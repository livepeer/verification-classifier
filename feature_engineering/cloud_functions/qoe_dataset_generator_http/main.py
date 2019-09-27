"""
Cloud function to generate transcoded renditions and quality metrics from
video sources.
It is invoked from the bash script "call_cloud_function.sh" that iteratively
triggers an http call for each video entry located in the designated bucket
"""
import subprocess

from os import makedirs, path, remove
from os.path import exists

from google.cloud import storage
from google.cloud import datastore
from google.api_core import retry

from urllib3.exceptions import ProtocolError

from imports import ffmpeg_installer

CODEC_TO_USE = 'libx264'

STORAGE_CLIENT = storage.Client()

SOURCES_BUCKET = 'livepeer-qoe-sources'
RENDITIONS_BUCKET = 'livepeer-qoe-renditions'


def check_blob(bucket_name, blob_name):
    """
    Checks if a file exists in the bucket.
    """

    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=blob_name).exists(STORAGE_CLIENT)

    print('File {} checked with status {}.'.format(blob_name, stats))
    return stats

def upload_blob(bucket_name, local_file, destination_blob_name):
    """
    Uploads a file to the bucket.
    """

    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print('Uploading {} to {}.'.format(
        local_file,
        destination_blob_name))

    blob.upload_from_filename(local_file)

    print('File {} uploaded to {}.'.format(
        local_file,
        destination_blob_name))

def download_to_local(bucket_name, local_folder, local_file, origin_blob_name):
    """
    Downloads a file from the bucket.
    """

    predicate = retry.if_exception_type(ConnectionResetError, ProtocolError)
    reset_retry = retry.Retry(predicate)

    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob('HD/{}'.format(origin_blob_name))
    print(origin_blob_name)
    print('File download Started…. Wait for the job to complete.')
    # Create this folder locally if not exists

    if not exists(local_folder):
        makedirs(local_folder)

    local_path = '{}/{}'.format(local_folder, local_file)
    print('Downloading {} to {}'.format(origin_blob_name, local_path))
    reset_retry(blob.download_to_filename(local_path))
    print('Downloaded {} to {}'.format(origin_blob_name, local_path))

def qoe_dataset_generator_http(request):
    """
    HTTP Cloud Function to generate video assets.
    Args:
        request: The request object, containing name and resolution
    Returns:
        The response text
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        source_name = request_json['name']
    elif request_args and 'name' in request_args:
        source_name = request_args['name']

    if request_json and 'resolution' in request_json:
        resolution = request_json['resolution']
    elif request_args and 'resolution' in request_args:
        resolution = request_args['resolution']

    ffmpeg_installer.install()

    print('Processing source: {} at resolution {}'.format(source_name, resolution))
    # Create the folder for the source asset
    source_folder = '/tmp/source'

    # Create the folder for the renditions
    renditions_folder = '/tmp/renditions'
    if not path.exists(renditions_folder):
        makedirs(renditions_folder)

    # Get the file that has been uploaded to GCS
    asset_path = {'path': '{}/{}'.format(source_folder, source_name)}

    # Check if the source is not already in the path
    if not path.exists(asset_path['path']):
        download_to_local(SOURCES_BUCKET, source_folder, source_name, source_name)

    qps = [45, 40, 32, 25, 21, 18, 14]

    print('Processing resolution', resolution)
    # Create folder for each rendition
    for qp_value in qps:

        bucket_path = '{}_{}/{}'.format(resolution, qp_value, source_name)
        check_blob(RENDITIONS_BUCKET, bucket_path)

        qp_path = '{}/{}_{}'.format(renditions_folder, resolution, qp_value)
        if not path.exists(qp_path):
            makedirs(qp_path)

    # Generate renditions with ffmpeg
    worker(asset_path['path'], CODEC_TO_USE, resolution, renditions_folder)

    #compute_metrics(asset_path, renditions_paths)

    # Upload renditions to GCE storage bucket
    for qp_value in qps:
        local_path = '{}/{}_{}/{}'.format(renditions_folder, resolution, qp_value, source_name)
        bucket_path = '{}_{}/{}'.format(resolution, qp_value, source_name)
        upload_blob(RENDITIONS_BUCKET, local_path, bucket_path)
        remove(local_path)

    return 'FINISHED Processing source: {} at resolution {}'.format(source_name, resolution)

def format_command(full_input_file, codec, resolution, output_folder):
    """
    Formats ffmpeg command to be executed in parallel for each Quantization parameter value
    """

    print('processing {}'.format(full_input_file))
    source_name = full_input_file.split('/')[-1]

    command = ['ffmpeg', '-y', '-i', '"{}"'.format(full_input_file),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 14', '"{}/{}_{}/{}"'.format(output_folder, resolution, '14', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 18', '"{}/{}_{}/{}"'.format(output_folder, resolution, '18', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 21', '"{}/{}_{}/{}"'.format(output_folder, resolution, '21', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 25', '"{}/{}_{}/{}"'.format(output_folder, resolution, '25', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 32', '"{}/{}_{}/{}"'.format(output_folder, resolution, '32', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 40', '"{}/{}_{}/{}"'.format(output_folder, resolution, '40', source_name),
               '-c:v', codec, '-vf', 'scale=-2:{}'.format(resolution), '-qp 45', '"{}/{}_{}/{}"'.format(output_folder, resolution, '45', source_name)
               ]
    return command


def worker(full_input_file, codec, bitrates, output_folder):
    """
    Executes ffmepg command via PIPE
    """

    ffmpeg_command = ''

    ffmpeg_command = format_command(full_input_file, codec, bitrates, output_folder)
    ffmpeg = subprocess.Popen(' '.join(ffmpeg_command),
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              shell=True)
    out, err = ffmpeg.communicate()
    print(' '.join(ffmpeg_command), out, err)
