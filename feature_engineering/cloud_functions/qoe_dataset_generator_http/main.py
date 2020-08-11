"""
Cloud function to generate transcoded renditions and quality metrics from
video sources.
It is invoked from the bash script "call_cloud_function.sh" that iteratively
triggers an http call for each video entry located in the designated bucket
"""
import sys
import subprocess
import datetime

import os

import time

import numpy as np

sys.path.insert(0, 'imports')

from imports import ffmpeg_installer
from imports import gce_utils

CODEC_TO_USE = 'libx264'

PARAMETERS_BUCKET = 'livepeer-qoe-renditions-params'
SOURCES_BUCKET = 'livepeer-qoe-sources'
RENDITIONS_BUCKET = 'livepeer-crf-renditions'
ENTITY_NAME = 'features_input_QoE'

def download_video_from_url(video_url, duration, local_file, extension):
    """
    Downloads a video from a given url to an HLS manifest
    """
    local_folder = os.path.dirname(local_file)
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    print('Downloading {} to {}'.format(video_url, local_file))
    seek_time = str(datetime.timedelta(seconds=int(duration)/2))
    end_time = str(datetime.timedelta(seconds=(int(duration)/2)+10))
    print(seek_time)
    ffmpeg_command = ['ffmpeg -y -i {} -ss {} -to {}'.format(video_url, seek_time, end_time),
                      '-vcodec copy',
                      '-acodec copy',
                      '-f {} {}'.format(extension, local_file)]

    ffmpeg = subprocess.Popen(' '.join(ffmpeg_command),
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              shell=True)
    out, err = ffmpeg.communicate()
    print(' '.join(ffmpeg_command), out, err)
    if not os.path.exists(local_file):
        print('Unable to download {}'.format(local_file))
        return False

    return True

def compute_metrics(asset, renditions):
    '''
    Function that instantiates the VideoAssetProcessor class with a list
    of metrics to be computed.
    The feature_list argument is left void as every descriptor of each
    temporal metric is potentially used for model training
    '''
    from imports.video_asset_processor import VideoAssetProcessor

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
                    ]

    print('Computing asset: {}, max samples used: {}'.format(asset, max_samples))
    asset_processor = VideoAssetProcessor(source_asset,
                                          renditions_list,
                                          metrics_list,
                                          False,
                                          max_samples,
                                          features_list=None)

    metrics_df, _, _ = asset_processor.video_capture()

    for _, row in metrics_df.iterrows():
        line = row.to_dict()
        for column in metrics_df.columns:
            if 'series' in column:
                line[column] = np.array2string(np.around(line[column], decimals=5))
        gce_utils.add_asset_input('{}/{}'.format(row['title'], row['attack']), line, ENTITY_NAME)

    elapsed_time = time.time() - start_time
    print('Computation time:', elapsed_time)

def dataset_generator_qoe_http(request):
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
    source_folder = '/tmp/{}'.format(os.path.dirname(source_name))
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    # Get the file that has been uploaded to GCS
    asset_path = {'path': '{}{}'.format(source_folder,
                                        source_name.replace(os.path.dirname(source_name), ''))}

    renditions_paths = []

    # Check if the source is not already in the path
    if not os.path.exists(asset_path['path']):
        gce_utils.download_to_local(SOURCES_BUCKET, asset_path['path'], source_name)

    #Bring the attacks to be processed locally
    resolutions = [1080, 720, 480, 384, 288, 144]
    crfs = [45, 40, 32, 25, 21, 18, 14]

    # Create a comprehension list with all the possible attacks
    rendition_list = ['{}_{}'.format(resolution, crf)
                      for resolution in resolutions
                      for crf in crfs
                      ]

    for rendition in rendition_list:
        remote_file = '{}/{}'.format(rendition, source_name)

        rendition_folder = '/tmp/{}'.format(rendition)
        local_path = '{}/{}'.format(rendition_folder, source_name)
        try:
            gce_utils.download_to_local(RENDITIONS_BUCKET,
                                        local_path,
                                        remote_file)

            renditions_paths.append({'path': local_path})

        except Exception as err:
            print('Unable to download {}/{}: {}'.format(rendition, source_name, err))

    if len(renditions_paths) > 0:
        print('Processing the following renditions: {}'.format(renditions_paths))
        compute_metrics(asset_path, renditions_paths)
    else:
        print('Empty renditions list. No renditions to process')

    # Cleanup 
    if os.path.exists(asset_path['path']):
        os.remove(asset_path['path'])
    for rendition in rendition_list:
        rendition_folder = '/tmp/{}'.format(rendition)
        local_path = '{}/{}'.format(rendition_folder, source_name)
        if os.path.exists(local_path):
            os.remove(local_path)
    return 'Process completed: {}'.format(asset_path['path'])


def trigger_renditions_bucket_event(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This function retrieves a source video and triggers
       the generation of renditions by means of an http asynchronous
       call to the create_renditions_http function

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None, the renditions cloud function are triggered asynchronously
    """

    name = data['name']

    # Create the folder for the renditions
    params_folder = '/tmp/{}'.format(os.path.dirname(name))
    if not os.path.exists(params_folder):
        os.makedirs(params_folder)

    resolutions = [1080, 720, 480, 384, 288, 144]
    crfs = [45, 40, 32, 25, 21, 18, 14]

    for resolution in resolutions:
        for crf in crfs:
            local_file = '{}/{}-{}-{}.json'.format(params_folder.replace(os.path.dirname(name), ''),
                                                   name,
                                                   resolution,
                                                   crf)
            remote_file = '{}/{}-{}.json'.format(name,
                                                 resolution,
                                                 crf)
            file_output = open(local_file, "w")
            file_output.close()
            gce_utils.upload_blob(PARAMETERS_BUCKET, local_file, remote_file)

    return 'Renditions triggered for {}'.format(name)

def create_renditions_bucket_event(data, context):
    """
    HTTP Cloud Function to generate video assets. Triggered by files
    deposited in PARAMETERS_BUCKET
    Args:
        data: The triggering object, containing name, resolution and quantization parameter
    Returns:
        The status message if successful
    """

    source_name = os.path.dirname(data['name'])
    params_name = data['name'].replace(source_name, '')
    resolution = params_name.split('-')[0][1:]
    crf_value = params_name.split('-')[1].replace('.json', '')

    print('Processing source: {} at resolution {}'.format(source_name, resolution))

    # Locate the ffmpeg binary
    ffmpeg_installer.install()

    # Create the folder for the source asset
    source_folder = '/tmp/source'

    # Create the folder for the renditions
    renditions_folder = '/tmp/renditions'
    if not os.path.exists(renditions_folder):
        os.makedirs(renditions_folder)

    # Get the file that has been uploaded to GCS
    asset_path = {'path': '{}/{}'.format(source_folder, source_name)}

    # Check if the source is not already in the path
    if not os.path.exists(asset_path['path']):
        print('Retrieving video from {}'.format(asset_path['path']))
        gce_utils.download_to_local(SOURCES_BUCKET, asset_path['path'], source_name)

    print('Processing resolution', resolution)
    # Create folder for each rendition

    bucket_path = '{}_{}/{}'.format(resolution, crf_value, source_name)
    print('Bucket path:', bucket_path)
    if not gce_utils.check_blob(RENDITIONS_BUCKET, bucket_path):
        crf_path = '{}/{}_{}/{}'.format(renditions_folder,
                                        resolution,
                                        crf_value,
                                        os.path.dirname(source_name))
        if not os.path.exists(crf_path):
            print('Creating rendition folder:', crf_path)
            os.makedirs(crf_path)

    # Generate renditions with ffmpeg
    renditions_worker(asset_path['path'],
                      source_folder,
                      CODEC_TO_USE,
                      resolution,
                      crf_value,
                      renditions_folder)

    #compute_metrics(asset_path, renditions_paths)

    # Upload renditions to GCE storage bucket

    local_path = '{}/{}_{}/{}'.format(renditions_folder, resolution, crf_value, source_name)
    bucket_path = '{}_{}/{}'.format(resolution, crf_value, source_name)
    gce_utils.upload_blob(RENDITIONS_BUCKET, local_path, bucket_path)
    os.remove(local_path)

    return 'FINISHED Processing source: {} at resolution {}'.format(source_name, resolution)

def renditions_worker(full_input_file, source_folder, codec, resolution, crf_value, output_folder):
    """
    Executes ffmepg command via PIPE
    """

    #Formats ffmpeg command to be executed in parallel for each Quantization parameter value
    print('processing {}'.format(full_input_file))
    source_name = full_input_file.replace('{}/'.format(source_folder), '')
    output_name = '"{}/{}_{}/{}"'.format(output_folder, resolution, crf_value, source_name)

    ffmpeg_command = ['ffmpeg', '-y', '-i', '"{}"'.format(full_input_file),
                      '-an',
                      '-c:v', codec,
                      '-copyts',
                      '-vsync 0',
                      '-copytb 1',
                      '-enc_time_base -1',
                      '-crf {}'.format(crf_value),
                      '-vf scale=-2:{}'.format(resolution),
                      output_name
                      ]

    ffmpeg = subprocess.Popen(' '.join(ffmpeg_command),
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              shell=True)
    out, err = ffmpeg.communicate()
    print(' '.join(ffmpeg_command), out, err)

def create_source_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The status message if successful
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json:
        playlist_url = request_json['playlist_url']
        video_id = request_json['video_id']
        extension = request_json['extension']
        duration = request_json['duration']

    elif request_args:
        playlist_url = request_args['playlist_url']
        video_id = request_args['video_id']
        extension = request_args['extension']
        duration = request_args['duration']
    else:
        return 'Unable to read request'
    print(playlist_url, video_id, extension)
    ffmpeg_installer.install()

    local_file = '/tmp/{}.{}'.format(video_id, extension)
    destination_blob_name = '{}.{}'.format(video_id, extension)

    if not gce_utils.check_blob(SOURCES_BUCKET, destination_blob_name):
        if download_video_from_url(playlist_url, duration, local_file, extension):
            gce_utils.upload_blob(SOURCES_BUCKET, local_file, destination_blob_name)
    else:
        print('Video already uploaded, skipping')
    return 'FINISHED Processing source: {}'.format(video_id)
