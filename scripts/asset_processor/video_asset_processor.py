"""
Module for management and parallelization of verification jobs.
"""

import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from collections import deque
import multiprocessing
from random import seed
from random import random

import cv2
import numpy as np
import pandas as pd
import logging
import math
from scipy.spatial import distance
from video_metrics import VideoMetrics

from ffmpeg_capture import VideoCapture

logger = logging.getLogger()


class VideoAssetProcessor:
	"""
	Class to extract and aggregate values from video sequences.
	It is instantiated as part of the data creation as well
	as in the inference, both in the CLI as in the notebooks.
	"""

	def __init__(self, original, renditions, metrics_list,
				 do_profiling=False, max_samples=-1, features_list=None, debug_frames=False, use_gpu=False):
		"""

		@param use_gpu:
		@param original:
		@param renditions:
		@param metrics_list:
		@param do_profiling:
		@param max_samples: Max number of matched master-rendition frames to calculate metrics against. -1 = all
		@param features_list:
		@param debug_frames: dump frames selected for metric extraction on disk, decreases performance
		"""
		# ************************************************************************
		# Initialize global variables
		# ************************************************************************

		self.debug_frames = debug_frames
		self.use_gpu = use_gpu
		# init debug dirs
		if self.debug_frames:
			self.frame_dir_name = type(self).__name__
			shutil.rmtree(self.frame_dir_name, ignore_errors=True)
			os.makedirs(self.frame_dir_name, exist_ok=True)
		if os.path.exists(original['path']):
			self.do_process = True
			self.original_path = original['path']
			self.master_capture = VideoCapture(self.original_path, use_gpu=use_gpu)
			# Frames Per Second of the original asset
			self.fps = self.master_capture.fps
			# Obtains number of frames of the original
			self.total_frames = self.master_capture.frame_count

			# Size of the hash for frame hash analysis in video_metrics
			self.hash_size = 16
			# Dictionary containing dict of metrics
			self.metrics = {}
			# List of metrics to be extracted from the asset and its renditions
			self.metrics_list = metrics_list
			# List of features to be extracted from the metrics list
			self.features_list = features_list
			# List of preverified renditions
			self.renditions_list = renditions

			if do_profiling:
				import line_profiler
				self.cpu_profiler = line_profiler.LineProfiler()
			else:
				self.cpu_profiler = None
			self.do_profiling = do_profiling

			# Check if HD list is necessary
			if 'temporal_ssim' in self.metrics_list or 'temporal_psnr' in self.metrics_list:
				self.make_hd_list = True
			else:
				self.make_hd_list = False
			# read renditions metadata like fps
			self.read_renditions_metadata()
			# Maximum number of frames to random sample
			if max_samples == -1:
				self.max_samples = self.total_frames
			else:
				# Take more master samples if rendition FPS is lower for a better chance to have good timestamp matches. This is a compromise between storing all master frames in memory or looping through all frames for each master-rendition pair.
				fps_ratios = [self.fps / r['fps'] for r in self.renditions_list]
				highest_ratio = np.max(fps_ratios)
				self.max_samples = max(max_samples, int(round(max_samples * highest_ratio)))
			# Convert OpenCV video captures of original to list
			# of numpy arrays for better performance of numerical computations
			self.master_indexes = []
			self.markup_master_frames = True
			master_idx_map, self.master_samples, self.master_samples_hd, self.master_pixels, self.height, self.width = self.capture_to_array(self.master_capture)
			self.master_capture.release()
			self.markup_master_frames = False
			# Instance of the video_metrics class
			self.video_metrics = VideoMetrics(self.metrics_list,
											  self.hash_size,
											  int(self.height),
											  self.cpu_profiler,
											  self.do_profiling)
			# Collects both dimensional values in a string
			self.dimensions = '{}:{}'.format(int(self.width), int(self.height))
			# Compute its features
			# Disable debug output for
			debug = self.debug_frames
			self.debug_frames = False
			self.metrics[self.original_path] = self.compute(master_idx_map,
															self.master_samples,
															self.master_samples_hd,
															self.original_path,
															self.dimensions,
															self.master_pixels)
			self.debug_frames = debug
		else:
			logger.error(f'Aborting, path does not exist: {original["path"]}')
			self.do_process = False

	def capture_to_array(self, capture):
		"""
		Function to convert OpenCV video capture to a list of
		numpy arrays for faster processing and analysis.
		@rtype: Tuple[list, np.ndarray, np.ndarray, int, int, int]
		@param capture:
		@return:  A tuple:
					- list mapping indexes in returned sample frames to corresponding master samples
					- sample frames
					- sample frames HD
					- total sample pixels count
					- original sample height
					- original sample width
		"""
		# Create list of random timestamps in video file to calculate metrics at
		if self.markup_master_frames:
			self.master_indexes = np.sort(np.random.choice(self.total_frames, self.max_samples, False))
			self.master_timestamps = []

		# difference between master timestamp and best matching frame timestamp of current video
		master_timestamp_diffs = [np.inf] * len(self.master_indexes)
		# currently selected frames
		candidate_frames = [None] * len(self.master_indexes)
		# maps selected rendition sample to master sample
		debug_index_mapping = {}
		master_idx_map = []
		frame_list = []
		frame_list_hd = []
		frames_read = 0
		pixels = 0
		height = 0
		width = 0
		timestamps_selected = []
		# Iterate through each frame in the video
		while True:
			# Read the frame from the capture
			frame_data = capture.read()
			if frame_data is not None:
				frames_read += 1
				if self.markup_master_frames:
					if frame_data.index in self.master_indexes:
						self.master_timestamps.append(frame_data.timestamp)
					else:
						continue
				# update candidate frames
				ts_diffs = [abs(frame_data.timestamp - mts) for mts in self.master_timestamps]
				best_match_idx = int(np.argmin(ts_diffs))
				best_match = np.min(ts_diffs)
				# max theoretical timestamp difference between 'matching' frames would be 1/(2*fps) + max(jitter)
				# don't consider frames that are too far, otherwise the algorithm will be linear on memory vs video length
				if best_match < 1 / (2 * capture.fps) and master_timestamp_diffs[best_match_idx] > best_match:
					master_timestamp_diffs[best_match_idx] = best_match
					candidate_frames[best_match_idx] = frame_data
			# Break the loop when frames cannot be taken from original
			else:
				break

		# process picked frames
		for i in range(len(candidate_frames)):
			frame_data = candidate_frames[i]
			ts_diff = master_timestamp_diffs[i]
			if frame_data is None or ts_diff > 1 / (2 * self.fps):
				# no good matching candidate frame
				continue
			if self.debug_frames:
				cv2.imwrite(f'{self.frame_dir_name}/{i:04}_{"m" if self.markup_master_frames else ""}_{frame_data.index}_{frame_data.timestamp:.4}.png', self._convert_debug_frame(frame_data.frame))
			timestamps_selected.append(frame_data.timestamp)
			master_idx_map.append(i)
			debug_index_mapping[self.master_indexes[i]] = frame_data.index
			# Count the number of pixels
			height = frame_data.frame.shape[1]
			width = frame_data.frame.shape[0]
			pixels += height * width

			# Change color space to have only luminance
			frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2HSV)[:, :, 2]
			frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_LINEAR)
			frame_list.append(frame)

			if self.make_hd_list:
				# Resize the frame
				if frame.shape[0] != 1920:
					frame_hd = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
				else:
					frame_hd = frame

				frame_list_hd.append(frame_hd)
		# Clean up memory
		capture.release()
		logger.info(f'Mean master-rendition timestamp diff, sec: {np.mean(list(filter(lambda x: not np.isinf(x), master_timestamp_diffs)))} SD: {np.std(list(filter(lambda x: not np.isinf(x), master_timestamp_diffs)))}')
		logger.info(f'Master frame index mapping for {capture.filename}: \n {debug_index_mapping}')
		return master_idx_map, np.array(frame_list), np.array(frame_list_hd), pixels, height, width

	@staticmethod
	def _convert_debug_frame(frame):
		return cv2.resize(frame, (1920, 1080), cv2.INTER_CUBIC)

	def compare_renditions_instant(self, rendition_sample_idx, master_sample_idx_map, frame_list, frame_list_hd, dimensions, pixels, path):
		"""
		Function to compare pairs of numpy arrays extracting their corresponding metrics.
		It basically takes the global original frame at frame_pos and its subsequent to
		compare them against the corresponding ones in frame_list (a rendition).
		It then extracts the metrics defined in the constructor under the metrics_list.
		Methods of comparison are implemented in the video_metrics class
		@param master_sample_idx_map: Mapping from rendition sample index to master sample index. If Nframes is different between master and rendition, the index mapping is not 1:1
		@param rendition_sample_idx: Index of master sample we compare rendition against
		@param frame_list:
		@param frame_list_hd:
		@param dimensions:
		@param pixels:
		@param path:
		@return:
		"""

		# Dictionary of metrics
		frame_metrics = {}
		# Original frame to compare against (downscaled for performance)
		reference_frame = self.master_samples[master_sample_idx_map[rendition_sample_idx]]
		# Original's subsequent frame (downscaled for performance)
		next_reference_frame = self.master_samples[master_sample_idx_map[rendition_sample_idx + 1]]
		# Rendition frame (downscaled for performance)
		rendition_frame = frame_list[rendition_sample_idx]
		# Rendition's subsequent frame (downscaled for performance)
		next_rendition_frame = frame_list[rendition_sample_idx + 1]
		if self.debug_frames:
			cv2.imwrite(f'{self.frame_dir_name}/CRI_{rendition_sample_idx:04}_ref.png', self._convert_debug_frame(reference_frame))
			cv2.imwrite(f'{self.frame_dir_name}/CRI_{rendition_sample_idx:04}_next_ref.png', self._convert_debug_frame(next_reference_frame))
			cv2.imwrite(f'{self.frame_dir_name}/CRI_{rendition_sample_idx:04}_rend.png', self._convert_debug_frame(rendition_frame))
			cv2.imwrite(f'{self.frame_dir_name}/CRI_{rendition_sample_idx:04}_next_rend.png', self._convert_debug_frame(next_rendition_frame))

		if self.make_hd_list:
			# Original frame to compare against (HD for QoE metrics)
			reference_frame_hd = self.master_samples_hd[rendition_sample_idx]
			# Rendition frame (HD for QoE metrics)
			rendition_frame_hd = frame_list_hd[rendition_sample_idx]

			# Compute the metrics defined in the global metrics_list.
			# Uses the global instance of video_metrics
			# Some metrics use a frame-to-frame comparison,
			# but other require current and forward frames to extract
			# their comparative values.
			rendition_metrics = self.video_metrics.compute_metrics(rendition_frame,
																   next_rendition_frame,
																   reference_frame,
																   next_reference_frame,
																   rendition_frame_hd,
																   reference_frame_hd)
		else:
			rendition_metrics = self.video_metrics.compute_metrics(rendition_frame,
																   next_rendition_frame,
																   reference_frame,
																   next_reference_frame)

		# Retrieve rendition dimensions for further evaluation
		rendition_metrics['dimensions'] = dimensions

		# Retrieve rendition number of pixels for further verification
		rendition_metrics['pixels'] = pixels

		# Retrieve rendition path for further identification
		rendition_metrics['ID'] = self.original_path

		# Identify rendition uniquely by its path and store metric data in frame_metrics dict
		frame_metrics[path] = rendition_metrics

		# Return the metrics, together with the position of the frame
		# frame_pos is needed for the ThreadPoolExecutor optimizations
		return rendition_metrics, rendition_sample_idx

	def compute(self, master_sample_idx_map, frame_list, frame_list_hd, path, dimensions, pixels):
		"""
		Function to compare lists of numpy arrays extracting their corresponding metrics.
		It basically takes the global original list of frames and the input frame_list
		of numpy arrrays to extract the metrics defined in the constructor.
		frame_pos establishes the index of the frames to be compared.
		It is optimized by means of the ThreadPoolExecutor of Python's concurrent package
		for better parallel performance.
		@param master_sample_idx_map: Mapping from rendition sample index to master sample index. If Nframes is different between master and rendition, the index mapping is not 1:1
		@param frame_list:
		@param frame_list_hd:
		@param path:
		@param dimensions:
		@param pixels:
		@return:
		"""

		# Dictionary of metrics
		rendition_metrics = {}
		# Position of the frame
		frame_pos = 0
		# List of frames to be processed
		frames_to_process = []

		# Iterate frame by frame and fill a list with their values
		# to be passed to the ThreadPoolExecutor. Stop when maximum
		# number of frames has been reached.

		frames_to_process = range(len(frame_list) - 1)

		# Execute computations in parallel using as many processors as possible
		# future_list is a dictionary storing all computed values from each thread
		with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
			# Compare the original asset against its renditions
			future_list = {executor.submit(self.compare_renditions_instant,
										   i,
										   master_sample_idx_map,
										   frame_list,
										   frame_list_hd,
										   dimensions,
										   pixels,
										   path): i for i in frames_to_process}

		# Once all frames in frame_list have been iterated, we can retrieve their values
		for future in future_list:
			# Values are retrieved in a dict, as a result of the executor's process
			result_rendition_metrics, frame_pos = future.result()
			# The computed values at a given frame

			rendition_metrics[frame_pos] = result_rendition_metrics

		# Return the metrics for the currently processed rendition
		return rendition_metrics

	def aggregate(self, metrics):
		"""
		Function to aggregate computed values of metrics and renditions into a
		pandas DataFrame.
		"""

		# Dictionary for containing all metrics
		metrics_dict = {}
		# Dictionary for containing all renditions
		renditions_dict = {}

		# Aggregate dictionary with all values for all renditions into a Pandas DataFrame
		# All values are stored and obtained in a per-frame basis, then in a per-rendition
		# fashion. They need to be rearranged.

		# First, we combine the frames
		dict_of_df = {k: pd.DataFrame(v) for k, v in metrics.items()}
		metrics_df = pd.concat(dict_of_df, axis=1, sort=True).transpose().reset_index(inplace=False)

		# Pandas concat function creates a level_0 and level_1 extra columns.
		# They need to be renamed
		metrics_df = metrics_df.rename(index=str,
									   columns={"level_1": "frame_num", "level_0": "path"})

		# Then we can combine each rendition
		for rendition in self.renditions_list:
			# For the current rendition, we need an empty dictionary
			rendition_dict = {}

			# We have a number of different metrics that have been computed.
			# These are an input for the constructor of the class an vary according to
			# what metrics are of interest in the research
			for metric in self.metrics_list:
				# Obtain a Pandas DataFrame from the original and build the original time series
				original_df = metrics_df[metrics_df['path'] == self.original_path][metric]
				original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)
				# Obtain a Pandas DataFrame from the current rendition and build its time series
				rendition_df = metrics_df[metrics_df['path'] == rendition['path']][metric]
				rendition_df = rendition_df.reset_index(drop=True)
				rendition_df = rendition_df.transpose().dropna().astype(float)

				# For those metrics that have a temporal character,
				# we need to make a further aggregation
				# We are basically using the Manhattan and euclidean distances,
				# and statistically meaningful
				# values such as mean, max and standard deviation.
				# The whole time series is also provided for later exploration
				#  in the analysis part.
				if 'temporal' in metric:
					x_original = np.array(original_df[rendition_df.index].values)
					x_rendition = np.array(rendition_df.values)

					[[manhattan]] = distance.cdist(x_original.reshape(1, -1),
												   x_rendition.reshape(1, -1),
												   metric='cityblock')

					rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original,
																					   x_rendition)
					rendition_dict['{}-manhattan'.format(metric)] = manhattan
					rendition_dict['{}-mean'.format(metric)] = np.mean(x_rendition)
					rendition_dict['{}-max'.format(metric)] = np.max(x_rendition)
					rendition_dict['{}-std'.format(metric)] = np.std(x_rendition)
					rendition_dict['{}-corr'.format(metric)] = np.correlate(x_original,
																			x_rendition,
																			mode='same').mean()
					rendition_dict['{}-series'.format(metric)] = x_rendition

				# Other metrics do not need time evaluation
				else:
					rendition_dict[metric] = rendition_df.mean()

			# Size is an important feature of an asset, as it gives important information
			# regarding the potential compression effect
			rendition_dict['size'] = os.path.getsize(rendition['path'])
			rendition_dict['fps'] = self.fps
			rendition_dict['path'] = rendition['path']

			# Extract the dimensions of the rendition
			dimensions_df = metrics_df[metrics_df['path'] == rendition['path']]['dimensions']
			rendition_dict['dimension'] = int(dimensions_df.unique()[0].split(':')[1])

			# Extract the pixels for this rendition
			pixels_df = metrics_df[metrics_df['path'] == rendition['path']]['pixels']
			rendition_dict['pixels'] = int(pixels_df.unique())

			# Store the rendition values in the dictionary of renditions for the present asset
			renditions_dict[rendition['path']] = rendition_dict

		# Add the current asset values to the global metrics_dict
		metrics_dict[self.original_path] = renditions_dict

		dict_of_df = {k: pd.DataFrame(v) for k, v in metrics_dict.items()}
		metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)

		pixels_df = metrics_df['pixels']

		# Compute a size/dimension ratio column for better accuracy
		metrics_df['size_dimension_ratio'] = metrics_df['size'] / metrics_df['dimension']

		metrics_df = self.cleanup_dataframe(metrics_df, self.features_list)

		metrics_df = metrics_df.drop(['dimension', 'size'], axis=1)

		return metrics_df, pixels_df, dimensions_df

	def cleanup_dataframe(self, metrics_df, features):
		"""
		Cleanup the resulting pandas dataframe and convert it to a numpy array
		to pass to the prediction model
		"""

		metrics_df = metrics_df.rename(columns={'level_0': 'title', 'level_1': 'attack'})

		if features is not None:
			if 'attack_ID' in features:
				features.remove('attack_ID')
			# Filter out features from metrics dataframe

			metrics_df = metrics_df[features]

			# Scale measured metrics according to their resolution for better accuracy
			metrics_df = self.rescale_to_resolution(metrics_df, features)

		return metrics_df

	@staticmethod
	def rescale_to_resolution(data, features):
		"""
		Function that improves model accuracy by scaling those features that
		"""
		feat_labels = ['dimension',
					   'size',
					   'fps',
					   'temporal_difference-euclidean',
					   'temporal_difference-manhattan',
					   'temporal_difference-max',
					   'temporal_difference-mean',
					   'temporal_difference-std',
					   'temporal_cross_correlation-euclidean',
					   'temporal_cross_correlation-manhattan',
					   'temporal_cross_correlation-max',
					   'temporal_cross_correlation-mean',
					   'temporal_cross_correlation-std',
					   'temporal_dct-euclidean',
					   'temporal_dct-manhattan',
					   'temporal_dct-max',
					   'temporal_dct-mean',
					   'temporal_dct-std',
					   'temporal_canny-euclidean',
					   'temporal_canny-manhattan',
					   'temporal_canny-max',
					   'temporal_canny-mean',
					   'temporal_canny-std',
					   'temporal_gaussian_mse-euclidean',
					   'temporal_gaussian_mse-manhattan',
					   'temporal_gaussian_mse-max',
					   'temporal_gaussian_mse-mean',
					   'temporal_gaussian_mse-std',
					   'temporal_gaussian_difference-euclidean',
					   'temporal_gaussian_difference-manhattan',
					   'temporal_gaussian_difference-max',
					   'temporal_gaussian_difference-mean',
					   'temporal_gaussian_difference-std',
					   'temporal_threshold_gaussian_difference-euclidean',
					   'temporal_threshold_gaussian_difference-manhattan',
					   'temporal_threshold_gaussian_difference-max',
					   'temporal_threshold_gaussian_difference-mean',
					   'temporal_threshold_gaussian_difference-std',
					   'temporal_histogram_distance-euclidean',
					   'temporal_histogram_distance-manhattan',
					   'temporal_histogram_distance-max',
					   'temporal_histogram_distance-mean',
					   'temporal_histogram_distance-std',
					   'temporal_ssim-euclidean',
					   'temporal_ssim-manhattan',
					   'temporal_ssim-max',
					   'temporal_ssim-mean',
					   'temporal_ssim-std',
					   'temporal_psnr-euclidean',
					   'temporal_psnr-manhattan',
					   'temporal_psnr-max',
					   'temporal_psnr-mean',
					   'temporal_psnr-std',
					   'temporal_entropy-euclidean',
					   'temporal_entropy-manhattan',
					   'temporal_entropy-max',
					   'temporal_entropy-mean',
					   'temporal_entropy-std',
					   'temporal_lbp-euclidean',
					   'temporal_lbp-manhattan',
					   'temporal_lbp-max',
					   'temporal_lbp-mean',
					   'temporal_lbp-std',
					   'temporal_orb-euclidean',
					   'temporal_orb-manhattan',
					   'temporal_orb-max',
					   'temporal_orb-mean',
					   'temporal_orb-std',
					   ]
		df_features = pd.DataFrame(data)
		downscale_features = ['temporal_cross_correlation'
							  ]

		upscale_features = ['temporal_difference',
							'temporal_dct',
							'temporal_canny',
							'temporal_gaussian_mse',
							'temporal_gaussian_difference',
							'temporal_histogram_distance',
							'temporal_entropy',
							'temporal_lbp'
							]

		for label in feat_labels:

			if label in features:
				if label.split('-')[0] in downscale_features:
					df_features[label] = df_features[label] / df_features['dimension']
					logger.debug(f'Downscaling {label}')
				elif label.split('-')[0] in upscale_features:
					df_features[label] = df_features[label] * df_features['dimension']
					logger.debug(f'Upscaling {label}')
		return df_features

	def process(self):
		"""
		Function to aggregate computed values of metrics
		of iterated renditions into a pandas DataFrame.
		"""
		if self.do_process:
			if self.do_profiling:
				self.capture_to_array = self.cpu_profiler(self.capture_to_array)
				self.compare_renditions_instant = self.cpu_profiler(self.compare_renditions_instant)
			# Iterate through renditions
			for rendition in self.renditions_list:
				path = rendition['path']
				capture = None
				try:
					if os.path.exists(path):
						capture = VideoCapture(path, use_gpu=self.use_gpu)
						# Turn openCV capture to a list of numpy arrays
						master_idx_map, frame_list, frame_list_hd, pixels, height, width = self.capture_to_array(capture)
						dimensions = '{}:{}'.format(int(width), int(height))
						# Compute the metrics for the rendition
						self.metrics[path] = self.compute(master_idx_map,
														  frame_list,
														  frame_list_hd,
														  path,
														  dimensions,
														  pixels)
					else:
						logger.error(f'Unable to find rendition file: {path}')
				except Exception as err:
					logger.exception('Unable to compute metrics for {}'.format(path))
				finally:
					if capture is not None:
						capture.release()

			if self.do_profiling:
				self.cpu_profiler.print_stats()
			if self.debug_frames:
				print(f'Frames metrics of {type(self).__name__}')
				print(self._convert_debug_metrics(self.metrics))
			return self.aggregate(self.metrics)
		else:
			logger.error('Unable to process. Original source path does not exist')
			return False

	@staticmethod
	def _convert_debug_metrics(metrics):
		res = []
		for f, frames in metrics.items():
			for id, frame in frames.items():
				row = {'file': f, 'sample': id}
				row.update(frame)
				res.append(row)
		df = pd.DataFrame(res)
		df.set_index(['file', 'sample'], inplace=True)
		df.sort_index(inplace=True)
		return df

	def read_renditions_metadata(self):
		# Iterate through renditions
		last_rendition_fps = None
		for rendition in self.renditions_list:
			path = rendition['path']
			try:
				if os.path.exists(path):
					capture = VideoCapture(path, use_gpu=False)
					# Get framerate
					fps = capture.fps
					# Validate frame rates, only renditions with same FPS (though not necessarily equal to source video) are currently supported in a single instance
					if last_rendition_fps is not None and last_rendition_fps != fps:
						raise Exception(f'Rendition has frame rate incompatible with other renditions: {fps}')
					rendition['fps'] = fps
					rendition['width'] = capture.width
					rendition['height'] = capture.height
				else:
					raise Exception(f'Rendition not found: {path}')
			finally:
				capture.release()
