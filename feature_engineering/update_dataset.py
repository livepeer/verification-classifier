'''
Script to update CSV dataset with new renditions data
'''
import pathlib
import random
import sys
import os
import shutil
import time
import numpy as np
import pandas as pd
import tqdm

pd.options.display.width = 0
pd.set_option('display.max_columns', None)
import argparse
import logging
from video_asset_processor import VideoAssetProcessor

# init logging
logging.basicConfig(level=logging.INFO,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def download_to_local(bucket_name, local_folder, local_file, origin_blob_name):
	"""
	Downloads a file from the bucket.
	"""

	predicate = retry.if_exception_type(ConnectionResetError, ProtocolError)
	reset_retry = retry.Retry(predicate)

	bucket = STORAGE_CLIENT.get_bucket(bucket_name)
	blob = bucket.blob('{}'.format(origin_blob_name))
	# print('Downloading blob {} from bucket {}'.format(origin_blob_name, bucket_name))
	# print('File download Started…. Wait for the job to complete.')
	# Create this folder locally if not exists
	if not os.path.exists(local_folder):
		os.makedirs(local_folder)

	local_path = '{}/{}'.format(local_folder, local_file)
	# print('Downloading {} to {}'.format(origin_blob_name, local_path))
	reset_retry(blob.download_to_filename(local_path))


# print('Downloaded {} to {}'.format(origin_blob_name, local_path))

def compute_metrics(asset, renditions):
	'''
	Function that instantiates the VideoAssetProcessor class with a list
	of metrics to be computed.
	The feature_list argument is left void as every descriptor of each
	temporal metric is potentially used for model training
	'''
	start_time = time.time()

	source_asset = asset

	max_samples = 30
	renditions_list = renditions
	metrics_list = ['temporal_ssim',
					'temporal_psnr',
					'temporal_dct',
					'temporal_gaussian_mse',
					'temporal_gaussian_difference',
					'temporal_threshold_gaussian_difference',
					'temporal_difference'
					]

	asset_processor = VideoAssetProcessor(source_asset,
										  renditions_list,
										  metrics_list,
										  False,
										  max_samples,
										  features_list=None)

	metrics_df, _, _ = asset_processor.process()

	for _, row in metrics_df.iterrows():
		line = row.to_dict()
		for column in metrics_df.columns:
			if 'series' in column:
				line[column] = np.array2string(np.around(line[column], decimals=5))
	return line


def add_asset_input(client, title, input_data):
	"""
	Function to add the asset's computed data to the database
	"""

	key = client.key(ENTITY_NAME, title, namespace='livepeer-verifier-QoE')
	video = datastore.Entity(key)

	video.update(input_data)

	client.put(video)


def dataset_generator_http(request):
	"""HTTP Cloud Function.
	Args:
		request (flask.Request): The request object, containing the name
		of the source asset
	Returns:
		The response text, or any set of values that can be turned into a
		Response object using `make_response`
	"""
	request_json = request.get_json(silent=True)
	request_args = request.args

	if request_json and 'name' in request_json:
		source_name = request_json['name']
		resolution_list = request_json['resolution_list'].split(',')
	elif request_args and 'name' in request_args:
		source_name = request_args['name']
		resolution_list = request_args['resolution_list'].split(',')
	print(resolution_list)
	# Create the folder for the source asset
	source_folder = '/tmp/1080p'
	# if not os.path.exists(source_folder):
	#     os.makedirs(source_folder)

	# Get the file that has been uploaded to GCS
	asset_path = {'path': '{}/{}'.format(source_folder, source_name)}

	renditions_paths = []

	# Check if the source is not already in the path
	if not os.path.exists(asset_path['path']):
		download_to_local(SOURCES_BUCKET, source_folder, source_name, source_name)

	# Bring the attacks to be processed locally
	# resolution_list = ['1080p', '720p', '480p', '360p', '240p', '144p']
	attack_names = ['watermark',
					'watermark-345x114',
					'watermark-856x856',
					'vignette',
					# 'rotate_90_clockwise',
					'black_and_white',
					'low_bitrate_4',
					'low_bitrate_8']

	# Create a comprehension list with all the possible attacks
	attacks_list = ['{}_{}'.format(resolution, attack)
					for resolution in resolution_list
					for attack in attack_names
					]
	if '1080p' in resolution_list:
		resolution_list.remove('1080p')
	attacks_list += resolution_list

	for attack in attacks_list:
		remote_file = '{}/{}'.format(attack, source_name)

		local_folder = '/tmp/{}'.format(attack)

		try:
			download_to_local(RENDITIONS_BUCKET,
							  local_folder,
							  source_name,
							  remote_file)

			local_file = '{}/{}'.format(local_folder, source_name)
			renditions_paths.append({'path': local_file})

		except Exception as err:
			print('Unable to download {}/{}: {}'.format(attack, source_name, err))

	if len(renditions_paths) > 0:
		print('Processing the following renditions: {}'.format(renditions_paths))
		compute_metrics(asset_path, renditions_paths)
	else:
		print('Empty renditions list. No renditions to process')

	# Cleanup
	if os.path.exists(asset_path['path']):
		os.remove(asset_path['path'])
	for rendition in attacks_list:
		rendition_folder = '/tmp/{}'.format(rendition)
		local_path = '{}/{}'.format(rendition_folder, source_name)
		if os.path.exists(local_path):
			os.remove(local_path)

	return 'Process completed: {}'.format(asset_path['path'])


def update_dataset(args):
	backup_path = args.dataset + '.bak'
	if not os.path.exists(backup_path) or os.path.getsize(backup_path) != os.path.getsize(args.dataset):
		shutil.copy(args.dataset, backup_path)
	rend_files = pathlib.Path(args.input).glob('**/*.*')
	# load data
	df = pd.read_csv(args.dataset)
	# generate id column
	if 'id' not in df.columns:
		df['id'] = df.attack.str.replace('/tmp/', '')
	df.index = df.id
	df.drop(axis=1, labels=['id'], inplace=True)
	i = 0
	# filter files
	rend_files_filtered = []
	for f in rend_files:
		rendition_id = os.sep.join(str(f).split(os.sep)[-2:])
		master_id = f'1080p{os.sep}{str(f).split(os.sep)[-1]}'
		master_path = args.input + os.sep + master_id
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
		rend_files_filtered.append(f)
	logger.info(f'Total master-renditions pairs to estimate metrics for: {len(rend_files_filtered)}')
	random.shuffle(rend_files_filtered)
	# compute metrics
	for f in tqdm.tqdm(rend_files_filtered):
		rendition_id = os.sep.join(str(f).split(os.sep)[-2:])
		master_id = f'1080p{os.sep}{str(f).split(os.sep)[-1]}'
		master_path = args.input + os.sep + master_id
		try:
			metrics = compute_metrics(dict(path=master_path), [dict(path=str(f))])
		except:
			logger.error(f'Error processing file {str(f)}, skipping...')
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
		i += 1
		if i % 500 == 0:
			df.to_csv(args.dataset)
	df.to_csv(args.dataset)



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input', help='Input folder with renditions. Metrics will be estimated only for files missing from the dataset.')
	ap.add_argument('-d', '--dataset', help='Dataset CSV to update. A backup will be created at the same location.')
	ap.add_argument('-f', '--filter', help='Filter for video files. All files without the filter substring in file or folder name will be excluded.')
	args = ap.parse_args()
	update_dataset(args)
