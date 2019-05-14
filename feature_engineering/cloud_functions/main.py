from google.cloud import storage
from google.cloud import datastore

import tarfile
import pickle
import time
import urllib
import numpy as np
import math
from scipy.spatial import distance
import cv2
import pandas as pd
import os
from concurrent.futures.thread import ThreadPoolExecutor
import datetime

import sys

sys.path.insert(0, 'imports')

from imports.video_asset_processor import video_asset_processor


storage_client = storage.Client()
datastore_client = datastore.Client()

def cli(asset, renditions):
    start_time = time.time()

    original_asset = asset

    renditions_list = renditions
    metrics_list = ['temporal_difference', 'temporal_canny', 'temporal_histogram_distance', 'temporal_cross_correlation', 'temporal_dct']

    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 4)

    metrics_df = asset_processor.process()

    X = np.asarray(metrics_df)

    for _,row in metrics_df.iterrows():
        line = row.to_dict()
        for column in metrics_df.columns:
            if 'series' in column:
                line[column] = np.array2string(np.around(line[column], decimals=5))
        add_asset_input(datastore_client,'{}/{}'.format(row['title'],row['attack']), line)

    elapsed_time = time.time() - start_time
    print('Prediction time:', elapsed_time)

def add_asset_input(client, title, input_data):
    entity_name = 'asset_input'
    key = client.key(entity_name, title, namespace = 'verifier-training')
    video = datastore.Entity(key)
    #input_data['created'] = datetime.datetime.utcnow()
    video.update(input_data)

    client.put(video)
def measure_asset_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args
  
    if request_json and 'name' in request_json:
        asset_name = request_json['name']
    elif request_args and 'name' in request_args:
        asset_name = request_args['name']

    original_bucket = storage_client.get_bucket('verifier-original')
    renditions_bucket = storage_client.get_bucket('verifier-renditions')
    
    # Get the file that has been uploaded to GCS
    asset_path = '/tmp/{}'.format(asset_name)
    print(asset_path)
    blob = original_bucket.get_blob(asset_name)
    blob.download_to_filename(asset_path)
    
    attacks_list = ['1080p_watermark',
                    '1080p_flip_vertical',
                    '1080p_rotate_90_clockwise',
                    '1080p_vignette',
                    '1080p_black_and_white',
                    '1080p_low_bitrate_4',
                    '720p',
                    '720p_vignette',
                    '720p_black_and_white',
                    '720p_low_bitrate_4',
                    '720p_watermark',
                    '720p_flip_vertical',
                    '720p_rotate_90_clockwise',
                    '480p',
                    '480p_watermark',
                    '480p_vignette',
                    '480p_black_and_white',
                    '480p_low_bitrate_4',
                    '480p_flip_vertical',
                    '480p_rotate_90_clockwise',
                    '360p',
                    '360p_watermark',
                    '360p_vignette',
                    '360p_black_and_white',
                    '360p_low_bitrate_4',
                    '360p_flip_vertical',
                    '360p_rotate_90_clockwise',
                    '240p',
                    '240p_watermark',
                    '240p_vignette',
                    '240p_black_and_white',
                    '240p_low_bitrate_4',
                    '240p_flip_vertical',
                    '240p_rotate_90_clockwise',
                    '144p',
                    '144p_watermark',
                    '144p_vignette',
                    '144p_black_and_white',
                    '144p_low_bitrate_4',
                    '144p_flip_vertical',
                    '144p_rotate_90_clockwise',
                    ]

    renditions_paths=[]
    for attack in attacks_list:
        remote_file = '{}/{}'.format(attack, asset_name)
        blob = renditions_bucket.get_blob(remote_file)
        
        local_folder = '/tmp/{}'.format(attack)
        local_file = '{}/{}'.format(local_folder, asset_name)
        
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        blob.download_to_filename(local_file)
        renditions_paths.append(local_file)

    cli(asset_path, renditions_paths)
