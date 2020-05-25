import requests
import json
import random
import numpy as np
import os
import glob
import verifier
import logging
from video_asset_processor_opencv import VideoAssetProcessorOpenCV
from video_asset_processor import VideoAssetProcessor
import timeit

logging.basicConfig(level=logging.INFO,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()


class Verifier:
	def __init__(self):
		self.ffmpeg_passes = 0
		self.ffmpeg_fails = 0
		self.opencv_passes = 0
		self.opencv_fails = 0

	def verify(self, in_file, out_file):
		url = "http://localhost:5000/verify"

		start = timeit.default_timer()
		res = verifier.verify(in_file, [{'uri': out_file}], False, -1, '../../models/', '', VideoAssetProcessorOpenCV, False, True)
		opencv_time = timeit.default_timer() - start
		tamper_opencv = float(res[0]["tamper"])

		start = timeit.default_timer()
		n_samples = -1
		gpu = False
		res = verifier.verify(in_file, [{'uri': out_file}], False, n_samples, '../../models/', '', VideoAssetProcessor, False, gpu)
		ffmpeg_time = timeit.default_timer() - start
		tamper_ffmpeg = float(res[0]["tamper"])

		logger.info(f'OpenCV processing took: {opencv_time} sec')
		logger.info(f'Ffmpeg processing with GPU={gpu} took: {ffmpeg_time} sec')
		logger.info(f"Tamper OpenCV: {tamper_opencv} Tamper Ffmpeg: {tamper_ffmpeg}")

		if tamper_ffmpeg < 0.0:
			logger.info("Failed!")
			self.ffmpeg_fails += 1
		else:
			logger.info("Passed!")
			self.ffmpeg_passes += 1

		if tamper_opencv < 0.0:
			logger.info("Failed!")
			self.opencv_fails += 1
		else:
			logger.info("Passed!")
			self.opencv_passes += 1

	def print_results(self, fails, passes):
		logger.info("Passes: {}".format(str(passes)))
		logger.info("Fails: {}".format(str(fails)))

		tpr = passes / (passes + fails)
		logger.info("TPR: {}".format(str(tpr)))


def run_test(source_dir, rendition_dir, files=None):
	verifier = Verifier()
	src_videos = sorted(glob.glob(source_dir + '/*'))
	for src in src_videos:
		filename = src.split(os.path.sep)[-1]
		if files is not None and not filename in files:
			continue
		rend_path = rendition_dir + os.path.sep + filename
		if not os.path.exists(rend_path):
			continue
		print('Videos:', src, rend_path)
		verifier.verify(src, rend_path)
	logger.info('FFMPEG RESULTS:')
	verifier.print_results(verifier.ffmpeg_fails, verifier.ffmpeg_passes)
	logger.info('OPENCV RESULTS:')
	verifier.print_results(verifier.opencv_fails, verifier.opencv_passes)


source_dir = '../../data/renditions/1080p/'
rendition_dir = '../../data/renditions/1080p_60-24fps_cpu/'
# files = ['pfkmHwfR8ms.mp4']
files = None

np.random.seed(123)
random.seed(123)
run_test(source_dir, rendition_dir, files)
