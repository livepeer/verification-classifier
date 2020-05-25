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

logging.basicConfig(level=logging.INFO,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()

class Verifier:
	def __init__(self):
		self.passes = 0
		self.fails = 0

	def verify(self, in_file, out_file):
		url = "http://localhost:5000/verify"

		res = verifier.verify(in_file, [{'uri': out_file}], False, 10, '../../models/', '', VideoAssetProcessorOpenCV)

		tamper = float(res[0]["tamper"])
		print("Tamper: {}".format(str(tamper)))

		if tamper < 0.0:
			print("Failed!")
			self.fails += 1
		else:
			print("Passed!")
			self.passes += 1

	def print_results(self):
		print("Passes: {}".format(str(self.passes)))
		print("Fails: {}".format(str(self.fails)))

		tpr = self.passes / (self.passes + self.fails)
		print("TPR: {}".format(str(tpr)))


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
	verifier.print_results()


source_dir = '../../data/renditions/1080p/'
rendition_dir = '../../data/renditions/720p/'
files = ['pfkmHwfR8ms.mp4']
#files = None

np.random.seed(123)
random.seed(123)
run_test(source_dir, rendition_dir, files)
