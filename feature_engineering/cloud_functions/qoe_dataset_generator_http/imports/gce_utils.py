"""
Module for managing GCE infrastructure
"""

import os

from google.api_core import retry

from urllib3.exceptions import ProtocolError

from google.cloud import datastore
from google.cloud import storage

DATASTORE_CLIENT = datastore.Client()
STORAGE_CLIENT = storage.Client()

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = STORAGE_CLIENT.list_blobs(bucket_name)
    blob_list = []
    for blob in blobs:
        blob_list.append(blob)

    return blob_list

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

def download_to_local(bucket_name, local_file, origin_blob_name):
    """
    Downloads a file from the bucket.
    """

    predicate = retry.if_exception_type(ConnectionResetError, ProtocolError)
    reset_retry = retry.Retry(predicate)

    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob(origin_blob_name)
    print(origin_blob_name)
    print('File download Started…. Wait for the job to complete.')
    # Create this folder locally if not exists
    local_folder = os.path.dirname(local_file)
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    print('Downloading {} to {}'.format(origin_blob_name, local_file))
    reset_retry(blob.download_to_filename(local_file))
    print('Downloaded {} to {}'.format(origin_blob_name, local_file))

def add_asset_input(title, input_data, entity_name):
    """
    Function to add the asset's computed data to the database
    """

    key = DATASTORE_CLIENT.key(entity_name, title, namespace='livepeer-verifier-QoE')
    video = datastore.Entity(key)

    video.update(input_data)

    DATASTORE_CLIENT.put(video)
