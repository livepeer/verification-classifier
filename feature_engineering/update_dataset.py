'''
Script to update CSV dataset with new renditions data
'''
import pathlib
import random
import re
import sys
import os
import shutil
import time
import numpy as np
import pandas as pd
import tqdm
import json
from machine_learning.cnn.image_dataset_generator import PairWriter

pd.options.display.width = 0
pd.set_option('display.max_columns', None)
import argparse
import logging
from scripts.asset_processor import video_asset_processor

# init logging
logging.basicConfig(level=logging.INFO,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()


# print('Downloaded {} to {}'.format(origin_blob_name, local_path))

def compute_metrics(asset, renditions, tamper, args):
	'''
	Function that instantiates the VideoAssetProcessor class with a list
	of metrics to be computed.
	The feature_list argument is left void as every descriptor of each
	temporal metric is potentially used for model training
	'''
	start_time = time.time()

	source_asset = asset

	max_samples = 10
	renditions_list = renditions
	metrics_list = ['temporal_dct',
					'temporal_gaussian_mse',
					'temporal_gaussian_difference',
					'temporal_threshold_gaussian_difference',
					'temporal_histogram_distance'
					]
	pair_callback = None
	if args.pairs:
		pv = PairWriter(args.pairs, tamper, (320, 320))
		pair_callback = pv.pair_callback

	asset_processor = video_asset_processor.VideoAssetProcessor(source_asset,
										  renditions_list,
										  metrics_list,
										  False,
										  max_samples,
										  features_list=None,
										  image_pair_callback=pair_callback)

	metrics_df, _, _ = asset_processor.process()
	if args.noarrays:
		array_cols = [c for c in metrics_df.columns if c.endswith('-series')]
		metrics_df.drop(array_cols, axis=1, inplace=True)
		return metrics_df.iloc[0]
	else:
		for _, row in metrics_df.iterrows():
			line = row.to_dict()
			for column in metrics_df.columns:
				if 'series' in column:
					line[column] = np.array2string(np.around(line[column], decimals=5))
			return line


def update_dataset(args):
	# backup_path = args.dataset + '.bak'
	# if not os.path.exists(backup_path) or os.path.getsize(backup_path) != os.path.getsize(args.dataset):
	# 	shutil.copy(args.dataset, backup_path)
	rend_files = pathlib.Path(args.input).glob('**/*.*')
	if os.path.exists(args.dataset):
		# load data
		df = pd.read_csv(args.dataset)
		# generate id column
		if 'id' not in df.columns:
			df['id'] = df.attack.str.replace('/tmp/', '')
	else:
		df = pd.DataFrame(columns=['id'])
	df.index = df.id
	df.drop(axis=1, labels=['id'], inplace=True)
	i = 0
	# filter files
	if args.filelist and os.path.exists(args.filelist):
		rend_files_filtered = pd.read_csv(args.filelist, header=None)[0].tolist()
		rend_files_filtered = list([r for r in rend_files_filtered if ':' not in r and r!=''])
	else:
		rend_files_filtered = []
		for f in tqdm.tqdm(rend_files, 'Generating video list'):
			rendition_id = os.sep.join(str(f).split(os.sep)[-2:])
			master_id = f'1080p{os.sep}{str(f).split(os.sep)[-1]}'
			master_path = args.originals + os.sep + master_id
			# skip if it's source video or source doesn't exist
			if master_id == rendition_id:
				continue
			if not os.path.exists(master_path):
				logger.warning(f'Source video doesn\'t exist: {master_path}')
				continue
			if rendition_id in df.index:
				continue
			if args.filter and args.filter not in rendition_id:
				continue
			rend_files_filtered.append(str(f))
	rend_files_filtered = list([f for f in rend_files_filtered if os.sep.join(f.split(os.sep)[-2:]) not in df.index])
	random.shuffle(rend_files_filtered)
	logger.info(f'Total master-renditions pairs to estimate metrics for: {len(rend_files_filtered)}')
	# compute metrics
	for f in tqdm.tqdm(rend_files_filtered, 'Processing video files'):
		rendition_id = os.sep.join(f.split(os.sep)[-2:])
		master_id = f'1080p{os.sep}{f.split(os.sep)[-1]}'
		master_path = args.originals + os.sep + master_id
		is_tamper = re.match('^[0-9]{3,4}p(_[0-9]+-[0-9]+fps)?(_gpu)?$', rendition_id.split(os.sep)[-2]) is None
		try:
			metrics = compute_metrics(dict(path=master_path), [dict(path=f)], is_tamper, args)
		except:
			logger.exception(f'Error processing file {f}, skipping...')
			continue
		metrics['attack'] = os.sep.join(metrics['attack'].split(os.sep)[-2:])
		metrics['title'] = os.sep.join(metrics['title'].split(os.sep)[-2:])
		metrics['path'] = os.sep.join(metrics['path'].split(os.sep)[-2:])
		values = []
		for novel_metric in set(metrics.keys()).difference(df.columns):
			df[novel_metric] = np.nan
		for c in df.columns:
			if c in metrics.keys():
				values.append(metrics[c])
			elif c == 'kind':
				values.append('features_input_60_540')
			elif c == 'Unnamed: 0':
				values.append(0)
			elif c == 'dimension':
				values.append(metrics['dimension_x'])
			else:
				raise Exception(f'Unknown column {c}')
		df.loc[rendition_id] = values
		if i == 0:
			# prettify column order
			cols_sorted = list(sorted(df.columns))
			cols_sorted.remove('title')
			cols_sorted = ['title'] + cols_sorted
			df = df[cols_sorted]
		i += 1
		if i % 500 == 0:
			df.to_csv(args.dataset)
	df.to_csv(args.dataset)



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input', help='Input folder with renditions. Metrics will be estimated only for files missing from the dataset.')
	ap.add_argument('-o', '--originals', help='Custom originals path')
	ap.add_argument('-d', '--dataset', help='Dataset CSV to update. A backup will be created at the same location.')
	ap.add_argument('-f', '--filter', help='Filter for video files. All files without the filter substring in file or folder name will be excluded.')
	ap.add_argument('-p', '--pairs', help='Path to image pairs dataset folder')
	ap.add_argument('-a', '--noarrays', help='Don\'t output arrays', action='store_true', default=False)
	ap.add_argument('-l', '--filelist', help='List of renditions files. Greatly speeds up process when using mounted storage bucket.')
	args = ap.parse_args()
	if not args.originals:
		args.originals = args.input
	update_dataset(args)
