"""
Cloud function to collect vimeo videos in a GCP bucket
with a given name.
It is triggered by means of an HTTP request
"""
import subprocess

from google.cloud import storage

from imports import ffmpeg_installer

STORAGE_CLIENT = storage.Client()
BUCKET_NAME = 'livepeer-qoe-sources'

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

def download_video(video_url, local_file, extension):
    """
    Downloads a video from a given url to an HLS manifest
    """
    bash_command = 'ffmpeg -i {} -vcodec copy -acodec copy -f {} {}'.format(video_url, extension, local_file)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)

    return True

def create_qoe_asset_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        A status message
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json:
        link = request_json['link']
        video_id = request_json['video_id']
        extension = request_json['extension']

    elif request_args:
        link = request_args['link']
        video_id = request_args['video_id']
        extension = request_args['extension']
    else:
        return 'Unable to read request'

    ffmpeg_installer.install()

    local_file = '/tmp/{}.{}'.format(video_id, extension)
    if download_video(link, local_file, extension):
        destination_blob_name = video_id
        upload_blob(BUCKET_NAME, local_file, destination_blob_name)

    return 'FINISHED Processing source: {}'.format(link)
