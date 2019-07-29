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

datastore_client = datastore.Client()

def compute_metrics(asset, renditions):
    start_time = time.time()

    original_asset = asset
    seconds = 1
    max_samples = 10
    renditions_list = renditions
    metrics_list = ['temporal_gaussian', 
                    'temporal_gaussian_difference', 
                    'temporal_dct'
                    ]

    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, seconds, max_samples, False)

    metrics_df = asset_processor.process()

    for _,row in metrics_df.iterrows():
        line = row.to_dict()
        for column in metrics_df.columns:
            if 'series' in column:
                line[column] = np.array2string(np.around(line[column], decimals=5))
        add_asset_input(datastore_client,'{}/{}'.format(row['title'],row['attack']), line)

    elapsed_time = time.time() - start_time
    print('Computation time:', elapsed_time)

def add_asset_input(client, title, input_data):
    entity_name = 'features_input'
    key = client.key(entity_name, title, namespace = 'livepeer-verifier-training')
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

    original_bucket = 'livepeer-verifier-originals'
    renditions_bucket = 'livepeer-verifier-renditions'
    
    # Create the folder for the original asset
    local_folder = '/tmp/1080p'
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # Get the file that has been uploaded to GCS
    asset_path = '{}/{}'.format(local_folder, asset_name)
    
    print(asset_path)
    renditions_paths=[]
    url = 'https://storage.googleapis.com/{}/{}'.format(original_bucket, asset_name)
    print('Downloading {}'.format(url))
    try:
        urllib.request.urlretrieve(url, asset_path)
        renditions_paths.append(asset_path)
    except:
        print('Unable to download {}'.format(url))
        pass

    attacks_list = ['1080p_watermark', 
                    '1080p_watermark-345x114', 
                    '1080p_watermark-856x856', 
                    # '1080p_vignette', 
                    # '1080p_flip_vertical',
                    # '1080p_rotate_90_clockwise',
                    # '1080p_black_and_white',
                    '1080p_low_bitrate_4',
                    '1080p_low_bitrate_8',
                    '720p',
                    '720p_watermark',
                    '720p_watermark-345x114',
                    '720p_watermark-856x856',
                    # '720p_vignette',
                    # '720p_black_and_white',
                    '720p_low_bitrate_4',
                    '720p_low_bitrate_8',
                    # '720p_flip_vertical',
                    # '720p_rotate_90_clockwise',
                    '480p',
                    '480p_watermark',
                    '480p_watermark-345x114',
                    '480p_watermark-856x856',
                    # '480p_vignette',
                    # '480p_black_and_white',
                    '480p_low_bitrate_4',
                    '480p_low_bitrate_8',
                    # '480p_flip_vertical',
                    # '480p_rotate_90_clockwise',
                    '360p',
                    '360p_watermark',
                    '360p_watermark-345x114',
                    '360p_watermark-856x856',
                    # '360p_vignette',
                    # '360p_black_and_white',
                    '360p_low_bitrate_4',
                    '360p_low_bitrate_8',
                    # '360p_flip_vertical',
                    # '360p_rotate_90_clockwise',
                    '240p',
                    '240p_watermark',
                    '240p_watermark-345x114',
                    '240p_watermark-856x856',
                    # '240p_vignette',
                    # '240p_black_and_white',
                    '240p_low_bitrate_4',
                    '240p_low_bitrate_8',
                    # '240p_flip_vertical',
                    # '240p_rotate_90_clockwise',
                    '144p',
                    '144p_watermark',
                    '144p_watermark-345x114',
                    '144p_watermark-856x856',
                    # '144p_vignette',
                    # '144p_black_and_white',
                    '144p_low_bitrate_4',
                    '144p_low_bitrate_8',
                    # '144p_flip_vertical',
                    # '144p_rotate_90_clockwise',
                    ]

    for attack in attacks_list:
        remote_file = '{}/{}'.format(attack, asset_name)
        url = 'https://storage.googleapis.com/{}/{}'.format(renditions_bucket, remote_file)
        
        local_folder = '/tmp/{}'.format(attack)
        local_file = '{}/{}'.format(local_folder, asset_name)
        
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        print('Downloading {}'.format(url))
        try:
            urllib.request.urlretrieve (url, local_file)
            renditions_paths.append(local_file)        
        except:
            print('Unable to download {}'.format(url))
            pass

    compute_metrics(asset_path, renditions_paths)
