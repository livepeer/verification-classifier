'''
Module wrapping up VideoAssetProcessor class in order to serve as interface for
CLI and API.
It manages pre-verification and tamper verfication of assets
'''

import uuid
import time
import json
import tarfile
import os
import sys
import urllib

import pickle
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import VideoAssetProcessor

def pre_verify(source_file, rendition):
    '''
    Function to verify that rendition conditions and specifications
    are met as prescribed by the Broadcaster
    '''
    # Extract data from video capture
    video_file = retrieve_video_file(rendition['uri'])
    rendition_capture = cv2.VideoCapture(video_file)
    fps = int(rendition_capture.get(cv2.CAP_PROP_FPS))

    # Create dictionary with passed / failed verification parameters

    rendition['path'] = video_file

    for key in rendition:
        if key == 'resolution':
            height = float(rendition_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = float(rendition_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            rendition['resolution']['height'] = height == float(rendition['resolution']['height'])
            rendition['resolution']['width'] = width == float(rendition['resolution']['width'])

        if key == 'frame_rate':
            rendition['frame_rate'] = fps == rendition['frame_rate']

        if key == 'bitrate':
            # Compute bitrate
            frame_count = int(rendition_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = float(frame_count) / float(fps) # in seconds
            bitrate = os.path.getsize(video_file) / duration
            rendition['bitrate'] = bitrate == rendition['bitrate']

    return rendition

def verify(source_uri, renditions, do_profiling, max_samples, model_dir, model_name):
    '''
    Function that returns the predicted compliance of a list of renditions
    with respect to a given source file using a specified model.
    '''

    total_start = time.clock()
    total_start_user = time.time()

    # Prepare source and renditions for verification
    original_asset = {'path':retrieve_video_file(source_uri),
                      'uri': source_uri}

    # Create a list of preverified renditions
    pre_verified_renditions = []
    for rendition in renditions:
        pre_verification = pre_verify(original_asset, rendition)
        pre_verified_renditions.append(pre_verification)

    # Configure model for inference
    model_name = 'OCSVM'
    scaler_type = 'StandardScaler'
    learning_type = 'UL'
    loaded_model = pickle.load(open('{}/{}.pickle.dat'.format(model_dir,
                                                              model_name), 'rb'))
    loaded_scaler = pickle.load(open('{}/{}_{}.pickle.dat'.format(model_dir,
                                                                  learning_type,
                                                                  scaler_type), 'rb'))

    # Open model configuration file
    with open('{}/param_{}.json'.format(model_dir, model_name)) as json_file:
        params = json.load(json_file)
        features = params['features']

    # Remove non numeric features from feature list
    non_temporal_features = ['attack_ID', 'title', 'attack', 'dimension', 'size']
    metrics_list = []
    for metric in features:
        if metric not in non_temporal_features:
            metrics_list.append(metric.split('-')[0])

    # Initialize times for assets processing profiling
    start = time.clock()
    start_user = time.time()

    # Instantiate VideoAssetProcessor class
    asset_processor = VideoAssetProcessor(original_asset,
                                          pre_verified_renditions,
                                          metrics_list,
                                          do_profiling,
                                          max_samples,
                                          features)

    # Record time for class initialization
    initialize_time = time.clock() - start
    initialize_time_user = time.time() - start_user

    # Register times for asset processing
    start = time.clock()
    start_user = time.time()

    # Assemble output dataframe with processed metrics
    metrics_df = asset_processor.process()

    # Record time for processing of assets metrics
    process_time = time.clock() - start
    process_time_user = time.time() - start_user

    # Normalize input data using the associated scaler
    x_renditions = np.asarray(metrics_df)
    x_renditions = loaded_scaler.transform(x_renditions)

    # Remove further features that model may not need
    matrix = pickle.load(open('{}/reduction_{}.pickle.dat'.format(model_dir, model_name), 'rb'))
    x_renditions = matrix.transform(x_renditions)

    # Make predictions for given data
    start = time.clock()
    y_pred = loaded_model.predict(x_renditions)
    prediction_time = time.clock() - start

    
    # Add predictions to rendition dictionary
    for i, rendition in enumerate(renditions):
        rendition.pop('path', None)
        rendition['tamper'] = int(y_pred[i])

    if do_profiling:
        print('Features used:', features)
        print('Total CPU time:', time.clock() - total_start)
        print('Total user time:', time.time() - total_start_user)
        print('Initialization CPU time:', initialize_time)
        print('Initialization user time:', initialize_time_user)

        print('Process CPU time:', process_time)
        print('Process user time:', process_time_user)
        print('Prediction CPU time:', prediction_time)

    return renditions

def retrieve_model(uri):
    '''
    Function to obtain pre-trained model for verification predictions
    '''

    model_dir = '/tmp/model'
    model_file = uri.split('/')[-1]
    # Create target Directory if don't exist
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created ")
        print('Model download started!')
        filename, _ = urllib.request.urlretrieve(uri,
                                                 filename='{}/{}'.format(model_dir,
                                                                         model_file)
                                                )
        print('Model downloaded')
        try:
            with tarfile.open(filename) as tar_f:
                tar_f.extractall(model_dir)
                return model_dir, model_file
        except Exception:
            return 'Unable to untar model'
    else:
        print("Directory ", model_dir, " already exists, skipping download")
        return model_dir, model_file

def retrieve_video_file(uri):
    '''
    Function to obtain a path to a video file from url or local path
    '''

    if 'http' in uri:
        file_name = '/tmp/{}'.format(uuid.uuid4())

        print('File download started!', flush=True)
        video_file, _ = urllib.request.urlretrieve(uri, filename=file_name)

        print('File downloaded', flush=True)
    else:
        video_file = uri
    return video_file
