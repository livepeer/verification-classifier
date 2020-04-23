"""
Module for management and parallelization of verification jobs.
"""

import os
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing
from random import seed
from random import random

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance

from video_metrics import VideoMetrics


class VideoAssetProcessor:
    """
    Class to extract and aggregate values from video sequences.
    It is instantiated as part of the data creation as well
    as in the inference, both in the CLI as in the notebooks.
    """
    def __init__(self, original, renditions, metrics_list,
                 do_profiling=False, max_samples=-1, features_list=None):
        # ************************************************************************
        # Initialize global variables
        # ************************************************************************

        # Stores system path to original asset
        if os.path.exists(original['path']):
            self.do_process = True
            self.original_path = original['path']
            # Initializes original asset to OpenCV VideoCapture class
            self.original_capture = cv2.VideoCapture(self.original_path)
            # Frames Per Second of the original asset
            self.fps = int(self.original_capture.get(cv2.CAP_PROP_FPS))
            # Obtains number of frames of the original
            self.max_frames = int(self.original_capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
            # Maximum number of frames to random sample
            if max_samples == -1:
                self.max_samples = self.max_frames
            else:
                self.max_samples = max_samples

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
            if 'ssim' in self.metrics_list or 'psnr' in self.metrics_list:
                self.make_hd_list = True
            else:
                self.make_hd_list = False

            # Convert OpenCV video captures of original to list
            # of numpy arrays for better performance of numerical computations
            self.random_sampler = []
            self.create_random_list = True
            self.original_capture, self.original_capture_hd, self.original_pixels, self.height, self.width = self.capture_to_array(self.original_capture)
            self.create_random_list = False
            # Instance of the video_metrics class
            self.video_metrics = VideoMetrics(self.metrics_list,
                                            self.hash_size,
                                            int(self.height),
                                            self.cpu_profiler,
                                            self.do_profiling)
            # Collects both dimensional values in a string
            self.dimensions = '{}:{}'.format(int(self.width), int(self.height))
            # Compute its features
            self.metrics[self.original_path] = self.compute(self.original_capture,
                                                            self.original_capture_hd,
                                                            self.original_path,
                                                            self.dimensions,
                                                            self.original_pixels)
        else:
            print('Aborting, original source not found in path provided')
            self.do_process = False
        
    def capture_to_array(self, capture):
        """
        Function to convert OpenCV video capture to a list of
        numpy arrays for faster processing and analysis
        """

        # List of numpy arrays
        frame_list = []
        frame_list_hd = []
        i = 0
        pixels = 0
        height = 0
        width = 0
        n_frame = 0
        # Iterate through each frame in the video
        while capture.isOpened():

            # Read the frame from the capture
            ret_frame, frame = capture.read()
            # If read successful, then append the retrieved numpy array to a python list
            if ret_frame:
                n_frame += 1
                add_frame = False

                if self.create_random_list:
                    random_frame = random()
                    if random_frame > 0.5:
                        add_frame = True
                        # Add the frame to the list if it belong to the random sampling list
                        self.random_sampler.append(n_frame)
                else:
                    if n_frame in self.random_sampler:
                        add_frame = True

                if add_frame:
                    i += 1
                    # Count the number of pixels
                    height = frame.shape[1]
                    width = frame.shape[0]
                    pixels += height * width

                    # Change color space to have only luminance
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
                    frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_LINEAR)
                    frame_list.append(frame)

                    if self.make_hd_list:
                        # Resize the frame 
                        if frame.shape[0] != 1920:
                            frame_hd = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                        else:
                            frame_hd = frame

                        frame_list_hd.append(frame_hd)
                    

                    if i > self.max_samples:
                        break

            # Break the loop when frames cannot be taken from original
            else:
                break
        # Clean up memory
        capture.release()            
        print(self.random_sampler, flush=True)
        return np.array(frame_list), np.array(frame_list_hd), pixels, height, width

    def compare_renditions_instant(self, frame_pos, frame_list, frame_list_hd, dimensions, pixels, path):
        """
        Function to compare pairs of numpy arrays extracting their corresponding metrics.
        It basically takes the global original frame at frame_pos and its subsequent to
        compare them against the corresponding ones in frame_list (a rendition).
        It then extracts the metrics defined in the constructor under the metrics_list.
        Methods of comparison are implemented in the video_metrics class
        """

        # Dictionary of metrics
        frame_metrics = {}
        # Original frame to compare against (downscaled for performance)
        reference_frame = self.original_capture[frame_pos]
        # Original's subsequent frame (downscaled for performance)
        next_reference_frame = self.original_capture[frame_pos+1]
        # Rendition frame (downscaled for performance)
        rendition_frame = frame_list[frame_pos]
        # Rendition's subsequent frame (downscaled for performance)
        next_rendition_frame = frame_list[frame_pos+1]

        if self.make_hd_list:
            # Original frame to compare against (HD for QoE metrics)
            reference_frame_hd = self.original_capture_hd[frame_pos]
            # Rendition frame (HD for QoE metrics)
            rendition_frame_hd = frame_list_hd[frame_pos]

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
        return rendition_metrics, frame_pos

    def compute(self, frame_list, frame_list_hd, path, dimensions, pixels):
        """
        Function to compare lists of numpy arrays extracting their corresponding metrics.
        It basically takes the global original list of frames and the input frame_list
        of numpy arrrays to extract the metrics defined in the constructor.
        frame_pos establishes the index of the frames to be compared.
        It is optimized by means of the ThreadPoolExecutor of Python's concurrent package
        for better parallel performance.
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

        frames_to_process = range(len(frame_list)-1)

        # Execute computations in parallel using as many processors as possible
        # future_list is a dictionary storing all computed values from each thread
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Compare the original asset against its renditions
            future_list = {executor.submit(self.compare_renditions_instant,
                                           i,
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
                    print('Downscaling', label, flush=True)
                elif label.split('-')[0] in upscale_features:
                    df_features[label] = df_features[label] * df_features['dimension']
                    print('Upscaling', label, flush=True)
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
                try:
                    if os.path.exists(path):
                        capture = cv2.VideoCapture(path)

                        # Turn openCV capture to a list of numpy arrays
                        frame_list, frame_list_hd, pixels, height, width = self.capture_to_array(capture)
                        dimensions = '{}:{}'.format(int(width), int(height))
                        # Compute the metrics for the rendition
                        self.metrics[path] = self.compute(frame_list,
                                                          frame_list_hd,
                                                          path,
                                                          dimensions,
                                                          pixels)
                    else:
                        print('Unable to find path')
                except Exception as err:
                    print('Unable to compute metrics for {}'.format(path))
                    print(err)

            if self.do_profiling:
                self.cpu_profiler.print_stats()

            return self.aggregate(self.metrics)
        else:
            print('Unable to process. Original source path does not exist')
            return False
