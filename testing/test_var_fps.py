"""
Module to test metrics robustness across renditions with different frame rates
"""
import sys
from functools import reduce
import logging
import time
import numpy as np
import ffmpeg_quality_metrics
import pandas as pd
import os
from random import seed

# init logging
logging.basicConfig(level=logging.DEBUG,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()

sys.path.insert(0, '../scripts/asset_processor/')

pd.options.display.width = 0
pd.set_option('display.max_columns', None)

from video_asset_processor import VideoAssetProcessor


def process(original_asset, renditions_list, metrics_list):
	seed(123)
	np.random.seed(123)
	asset_processor = VideoAssetProcessor(original_asset,
										  renditions_list,
										  metrics_list,
										  do_profiling=False,
										  max_samples=10
										  )
	data_df, pixels_df, dimensions_df = asset_processor.process()
	return data_df


def get_metrics():
	original_asset = {'path': '../../data/renditions/1080p/0vFdsx2x-wI.mp4'}
	renditions_list = [{'path': '../../data/renditions/1080p_watermark_60-30fps/0vFdsx2x-wI.mp4'}
					   ]
	metrics_list = ['temporal_ssim',
					'temporal_psnr']

	logger.info('Processing videos...')
	data_df = process(original_asset, renditions_list, metrics_list)
	logger.info('Done')

	print(data_df)


if __name__ == '__main__':
	get_metrics()
