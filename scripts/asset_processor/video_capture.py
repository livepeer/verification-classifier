"""
Ffmpeg-based video file reader with timestamp support and optional GPU decoding
"""
import os
import re
import time
from typing import Union, Tuple
import numpy as np
import bisect
import subprocess
import threading
import logging

logger = logging.getLogger()


class YUV2RGB_GPU():
	"""
	High performance YUV - RGB conversion with Tensorflow
	"""

	def __init__(self, w=1920, h=1080):
		config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.03))
		self.y = tf.placeholder(shape=(1, h, w), dtype=tf.float32)
		self.u = tf.placeholder(shape=(1, h, w), dtype=tf.float32)
		self.v = tf.placeholder(shape=(1, h, w), dtype=tf.float32)
		r = self.y + 1.371 * (self.v - 128)
		g = self.y + 0.338 * (self.u - 128) - 0.698 * (self.v - 128)
		b = self.y + 1.732 * (self.u - 128)
		result = tf.stack([b, g, r], axis=-1)
		self.result = tf.clip_by_value(result, 0, 255)
		self.sess = tf.Session(config=config)

	def convert(self, y, u, v):
		results = self.sess.run(self.result, feed_dict={self.y: y, self.u: u, self.v: v})
		return results.astype(np.uint8)


class VideoCapture:
	# how many times to poll for timestamp availability before generating error
	MAX_TIMESTAMP_WAIT = 100
	TIMESTAMP_POLL_INTERVAL = 0.01

	class FrameData:
		"""
		Object holding pixel data and metadata
		"""

		def __init__(self, index: int, timestamp: float, frame: np.ndarray):
			self.frame = frame
			self.index = index
			self.timestamp = timestamp

	def __init__(self, filename: str, use_gpu=False, video_reader: str = 'opencv'):
		"""

		@param filename:
		@param use_gpu:
		@param video_reader: 'ffmpeg_bgr' - read video with ffmpeg bgr24 output, warning: Ffmpeg has some color conversion issue which adds irregular noise to pixel data
							 'ffmpeg_yuv' - read video with ffmpeg yuv420p output, slower, requires tensorflow
							 'opencv' - read video with opencv, and use ffmpeg only for reading timestamps, fastest, but scans video 2 times
		"""
		if not os.path.exists(filename):
			raise ValueError(f'File {filename} doesn\'t exist')
		if video_reader not in ['ffmpeg_bgr', 'ffmpeg_yuv', 'opencv']:
			raise ValueError(f'Unknown video reader type {video_reader}')
		logger.info(f'Video reader is: {video_reader}')
		if video_reader == 'ffmpeg_yuv':
			global tf
			import tensorflow as tf
			self.pixel_converter = YUV2RGB_GPU(self.width, self.height)
		self.video_reader = video_reader
		self.ts_reader_thread: threading.Thread
		self.filename = filename
		self.started = False
		self.stopping = False
		self.timestamps = []
		self.frame_idx = -1
		self.stream_data_read = False
		self.ffmpeg_decoder = ''
		self.offset = 0
		self.opencv_capture = None
		self.last_frame_data = None
		if use_gpu:
			# Nvcodec sometimes duplicates frames producing more frames than it\'s actually in the video. In tests, it happened only at the end of the video, but potentially it can corrupt timestamps
			self.ffmpeg_decoder = '-hwaccel nvdec -c:v h264_cuvid'
			# this enables GPU codec for Ffmpeg libs used by OpenCV
			if 'opencv' in video_reader:
				os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;h264_cuvid'
				logger.warning('For OpenCV+Ffmpeg GPU acceleration to work, config environment variable must be set before the first cv2 import')
		if 'opencv' in video_reader:
			global cv2
			import cv2
			if not (cv2.getVersionMajor() >= 4 and cv2.getVersionMinor() >= 2):
				raise Exception('Can\'t use OpenCV to read video - minimum required version of opencv-python is 4.2')

		self._read_metadata()

	def _read_metadata(self):
		"""
		Reads video properties and fills corresponding fields
		@return:
		"""
		cap = None
		try:
			cap = cv2.VideoCapture(self.filename, cv2.CAP_FFMPEG)
			self.fps = cap.get(cv2.CAP_PROP_FPS)
			self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			if self.video_reader == 'opencv':
				self.opencv_capture = cap
			logger.info(f'Video file opened {self.filename}, {self.width}x{self.height}, {self.fps} FPS')
		finally:
			if cap is not None and self.video_reader != 'opencv':
				cap.release()

	def _read_next_frame(self, grab=False):
		if self.video_reader == 'ffmpeg_yuv':
			# get raw frame from stdout and convert it to numpy array
			bytes = self.video_capture.stdout.read(int(self.height * self.width * 6 // 4))
			if len(bytes) == 0:
				return None
			k = self.height * self.width
			y = np.frombuffer(bytes[0:k], dtype=np.uint8).reshape((self.height, self.width))
			u = np.frombuffer(bytes[k:k + k // 4], dtype=np.uint8).reshape((self.height // 2, self.width // 2))
			v = np.frombuffer(bytes[k + k // 4:], dtype=np.uint8).reshape((self.height // 2, self.width // 2))
			u = np.reshape(cv2.resize(np.expand_dims(u, -1), (self.width, self.height)), (self.height, self.width))
			v = np.reshape(cv2.resize(np.expand_dims(v, -1), (self.width, self.height)), (self.height, self.width))
			return self.pixel_converter.convert([y], [u], [v])[0]
		elif self.video_reader == 'ffmpeg_bgr':
			bytes = self.video_capture.stdout.read(int(self.height * self.width * 3))
			if len(bytes) == 0:
				return None
			return np.frombuffer(bytes, np.uint8).reshape([self.height, self.width, 3])
		elif self.video_reader == 'opencv':
			if not grab:
				return self.opencv_capture.read()[1]
			else:
				return self.opencv_capture.grab()

	def read(self, grab=False) -> Union[FrameData, None]:
		"""
		Reads next frame from video.
		@param grab: Works for OpenCV reader only. If true, doesn't decode the frame, it will be empty in FrameData object. Use retrieve() to get frame data.
		@return:
		@return: Tuple[frame_index, frame_timestamp, frame] or [None, None, None] if end of video
		"""
		if not self.started:
			self.start()
		frame = self._read_next_frame(grab)
		if frame is None or (grab and frame == False):
			return None
		self.frame_idx += 1
		if 0 < self.frame_count == self.frame_idx:
			logger.error(f'Frame count mismatch, possibly corrupted video file: {self.filename}')
			self.release()
			return None
		timestamp = self._get_timestamp_for_frame(self.frame_idx)
		logger.debug(f'Read frame {self.frame_idx} at PTS_TIME {timestamp}')
		self.last_frame_data = VideoCapture.FrameData(self.frame_idx, timestamp, frame)
		return self.last_frame_data

	def retrieve(self):
		if self.video_reader == 'opencv':
			self.last_frame_data.frame = self.opencv_capture.retrieve()[1]
		return self.last_frame_data

	def _get_timestamp_for_frame(self, frame_idx) -> float:
		if self.video_reader == 'opencv':
			# opencv handles offset internally
			opencv_ts = self.opencv_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
			self.timestamps.append(opencv_ts)
		else:
			# wait for timestamp record to be available, normally it available before frame is read
			waits = 0
			while frame_idx > len(self.timestamps) - 1:
				time.sleep(VideoCapture.TIMESTAMP_POLL_INTERVAL)
				waits += 1
				if waits > VideoCapture.MAX_TIMESTAMP_WAIT:
					raise Exception('Error reading video timestamps')
			if waits > 0:
				logger.debug(f'Waited for frame timestamp for {VideoCapture.TIMESTAMP_POLL_INTERVAL * waits} sec')
		return self.timestamps[frame_idx]

	def start(self):
		if self.video_reader != 'opencv':
			format = 'null'
			pix_fmt = 'yuv420p' if self.video_reader == 'ffmpeg_yuv' else ('bgr24' if self.video_reader == 'ffmpeg_bgr' else '')
			if pix_fmt:
				pix_fmt = f'-pix_fmt {pix_fmt}'
				format = 'rawvideo'
			output = 'pipe:' if self.video_reader != 'opencv' else '-'
			# start ffmpeg process
			ffmpeg_cmd = f"ffmpeg -y -debug_ts -hide_banner {self.ffmpeg_decoder} -i {self.filename} -copyts -f {format} {pix_fmt} {output}"
			self.video_capture = subprocess.Popen(ffmpeg_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			# stderr and stdout are not synchronized, read timestamp data in separate thread
			self.ts_reader_thread = threading.Thread(target=self.stream_reader, args=[self.video_capture.stderr])
			self.ts_reader_thread.start()
			# wait for stream reader thread to fill timestamp list
			time.sleep(0.05)
		self.started = True

	def stream_reader(self, stream):
		while not self.stopping:
			try:
				last_line = stream.readline().decode('ascii')
				if not last_line:
					break
				if not self.stream_data_read:
					# read stream offset
					m = re.match('.+Duration:.+start: (?P<start>\d*\.?\d*)', last_line)
					if m:
						self.offset = float(m.group('start'))
						logger.info(f'Video start offset is: {self.offset}')
						self.stream_data_read = True
				m = re.match('^demuxer\+ffmpeg -> ist_index:[0-9].+type:video.+pkt_pts_time:(?P<pkt_pts_time>\d*\.?\d*)', last_line)
				if m:
					timestamp = float(m.group('pkt_pts_time'))
					if timestamp < self.offset:
						logger.warning('Unknown behavior: pkt_pts_time is expected to be greater than stream start offset')
						timestamp = self.offset
					# Some frames are out-of-order by PTS, but returned to output in proper order. This may fail if corresponding debug record wasn't yet fetched when frame was read, but such behavior never observed during testing.
					bisect.insort(self.timestamps, timestamp - self.offset)
					if 2 < len(self.timestamps) < self.frame_idx + 3:
						logger.warning('Don\'t have enough timestamp records to account for out-of-order frames')
					self.timestamps = list(sorted(self.timestamps))
					if not self.stream_data_read:
						# stream data wasn't parsed, no point in searching for it
						logger.warning('Unable to parse stream data, start offset set to 0')
						self.stream_data_read = True
			except:
				if not self.stopping:
					raise

	def release(self):
		"""
		Stop Ffmpeg instance
		@return:
		"""
		try:
			if self.started:
				self.stopping = True
				self.video_capture.terminate()
				if self.opencv_capture is not None:
					self.opencv_capture.release()
		except:
			pass
