"""
Module for iterative creation of params files that trigger rendition creation
"""

import main
from imports import gce_utils

SOURCES_BUCKET = 'livepeer-qoe-sources'

def upload_renditions_params():
    blob_list = gce_utils.list_blobs(SOURCES_BUCKET)
    print('Iterating blob list')
    context = ''

    for blob in blob_list:
        try:
            source = {'name': blob.name}
            main.trigger_renditions_bucket_event(source, context)
        except Exception as err:
            print('Error {} happened in {}'.format(str(err), blob.name))
upload_renditions_params()
