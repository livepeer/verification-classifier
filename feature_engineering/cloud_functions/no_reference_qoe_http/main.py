"""
Module for generation of brisque data obtained from processing a set of videos 
stored in a GCE bucket.
It is deployed as a cloud function.
"""

import cv2
import numpy as np
import pandas as pd
import time
from google.cloud import datastore

DATASTORE_CLIENT = datastore.Client()
ENTITY_NAME = 'features_input_brisque'

def dataset_generator_brisque_http(request):
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

    source_url = 'https://storage.googleapis.com/livepeer-qoe-sources/vimeo/{}'.format(source_name)
    print(source_name)
    cap = cv2.VideoCapture(source_url)

    start_time = time.time()
    i = 0
    n_frame = 0
    metrics_dict = {}
    while cap.isOpened():
        ret, frame = cap.read()
        n_frame += 1
        if ret:
            i += 1
            if i == 5:
                features = np.empty([36,])
                features = cv2.quality.QualityBRISQUE_computeFeatures(frame, features)
                features = np.array2string(np.around(features, decimals=5))
                metrics_dict[str(n_frame)] = features
                
                i = 0
        else:
            break
    print(metrics_dict)
    add_asset_input(source_name, metrics_dict, ENTITY_NAME)
    return 'Process completed: {} took {}s'.format(source_name, time.time() - start_time)

def add_asset_input(title, input_data, entity_name):
    """
    Function to add the asset's computed data to the database
    """

    key = DATASTORE_CLIENT.key(entity_name, title, namespace='livepeer-verifier-brisque')
    video = datastore.Entity(key)

    video.update(input_data)

    DATASTORE_CLIENT.put(video)
