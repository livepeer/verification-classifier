import requests
import json
import random
import numpy as np
import pandas as pd
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

		# start = timeit.default_timer()
		# res = verifier.verify(in_file, [{'uri': out_file}], False, -1, '../../models/', '', VideoAssetProcessorOpenCV, False, True)
		# opencv_time = timeit.default_timer() - start
		# tamper_opencv = float(res[0]["tamper"])
		tamper_opencv = -1
		opencv_time = 0

		start = timeit.default_timer()
		n_samples = 10
		gpu = False
		res = verifier.verify(in_file, [{'uri': out_file}], False, n_samples, '../../models/', '', VideoAssetProcessor, False, gpu)
		ffmpeg_time = timeit.default_timer() - start
		tamper_ffmpeg = float(res[0]["tamper"])

		return {'score': tamper_ffmpeg, 'time_sec': ffmpeg_time}

	def print_results(self, fails, passes):
		logger.info("Passes: {}".format(str(passes)))
		logger.info("Fails: {}".format(str(fails)))

		tpr = passes / (passes + fails)
		logger.info("TPR: {}".format(str(tpr)))


def run_test(source_dir, rendition_dirs, files=None):
	verifier = Verifier()
	src_videos = sorted(glob.glob(source_dir + '/*'))
	results = []
	for src in src_videos:
		filename = src.split(os.path.sep)[-1]
		if files is not None and not filename in files:
			continue
		i = 0
		for rendition_dir, tamper in rendition_dirs:
			rendition_name = rendition_dir.strip(os.path.sep).split(os.path.sep)[-1]
			rend_path = rendition_dir + os.path.sep + filename
			if not os.path.exists(rend_path):
				continue
			res = verifier.verify(src, rend_path)
			res['master_filename'] = filename
			res['rendition_type'] = rendition_name
			res['is_tamper'] = tamper
			results.append(res)
	df_res: pd.DataFrame = pd.DataFrame(results)
	df_res.set_index(['master_filename', 'rendition_type'], inplace=True)
	df_res.sort_index(inplace=True)
	df_res.to_csv('../feature_engineering/notebooks/test_fps_renditions.csv')
	print(df_res)


source_dir = '../../data/renditions/1080p/'
rendition_dirs = [
	('../../data/renditions/720p/', False),
	('../../data/renditions/1080p_60-30fps_cpu_cpr/', False),
	('../../data/renditions/1080p_60-24fps_cpu_cpr/', False),
	('../../data/renditions/1080p_watermark_60-30fps_cpu_cpr/', True),
	('../../data/renditions/720p_60-30fps_cpu_cpr/', False),
	('../../data/renditions/720p_watermark_60-30fps_cpu_cpr/', True)
]
files = None
# files = ['pfkmHwfR8ms.mp4']
# files = ['0fIdY5IAnhY.mp4']


np.random.seed(123)
random.seed(123)
run_test(source_dir, rendition_dirs, files)
