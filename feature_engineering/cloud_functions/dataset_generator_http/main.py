'''
Main function to be called from GCE's cloud function
This function is in charge of adding training data to
the datastore for later generation of models and feature study
'''

import sys
import os
import time
import numpy as np

from google.cloud import datastore
from google.cloud import storage
from google.api_core import retry

from urllib3.exceptions import ProtocolError

sys.path.insert(0, 'imports')

from imports.video_asset_processor import VideoAssetProcessor

DATASTORE_CLIENT = datastore.Client()
STORAGE_CLIENT = storage.Client()

SOURCES_BUCKET = 'livepeer-verifier-originals'
RENDITIONS_BUCKET = 'livepeer-verifier-renditions'
ENTITY_NAME = 'features_input_60_540'

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
                    'temporal_brisque'
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
        add_asset_input(DATASTORE_CLIENT, '{}/{}'.format(row['title'], row['attack']), line)

    elapsed_time = time.time() - start_time
    print('Computation time:', elapsed_time)


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
    elif request_args and 'name' in request_args:
        source_name = request_args['name']

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

    #Bring the attacks to be processed locally
    resolution_list = ['1080p', '720p', '480p', '360p', '240p', '144p']
    attack_names = ['watermark',
                    'watermark-345x114',
                    'watermark-856x856',
                    'vignette',
                    'rotate_90_clockwise',
                    'black_and_white',
                    'low_bitrate_4',
                    'low_bitrate_8']

    # Create a comprehension list with all the possible attacks
    attacks_list = ['{}_{}'.format(resolution, attack)
                    for resolution in resolution_list
                    for attack in attack_names
                    ]

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
