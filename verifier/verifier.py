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


def verify(asset, renditions, do_profiling, max_samples, model_dir, model_name):
    seconds = 2
    
    total_start = time.clock()
    total_start_user = time.time()

    # Configure model for inference
    model_name = 'OCSVM'
    scaler_type = 'StandardScaler'
    learning_type = 'UL'
    loaded_model = pickle.load(open('{}/{}.pickle.dat'.format(model_dir, model_name), 'rb'))
    loaded_scaler = pickle.load(open('{}/{}_{}.pickle.dat'.format(model_dir, learning_type, scaler_type), 'rb'))
    # Open model configuration file
    with open('{}/param_{}.json'.format(model_dir, model_name)) as json_file:
        params = json.load(json_file)
        features = params['features']

    # Prepare input and renditions for verification
    original_asset = asset
    renditions_list = list(renditions)

    
    # Remove non numeric features from feature list
    non_numeric_features = ['attack_ID', 'title', 'attack', 'dimension', 'size']
    metrics_list = []
    for metric in features:
        if metric not in non_numeric_features:
            metrics_list.append(metric.split('-')[0])
    print(features, metrics_list)
    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    start_user = time.time()
    
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, seconds, max_samples, do_profiling)
    initialize_time = time.clock() - start
    initialize_time_user = time.time() - start_user
    
    start = time.clock()
    start_user = time.time()
    
    # Assemble output dataframe
    metrics_df = asset_processor.process()
    
    process_time = time.clock() - start
    process_time_user = time.time() - start_user

    # Cleanup the resulting pandas dataframe and convert it to a numpy array
    # to pass to the prediction model
    for column in metrics_df.columns:
        if 'series' in column:
            metrics_df = metrics_df.drop([column], axis=1)

    features.remove('attack_ID')

    metrics_df = metrics_df[features]
    metrics_df = metrics_df.drop('title', axis=1)
    metrics_df = metrics_df.drop('attack', axis=1)

    metrics_df = rescale_to_resolution(metrics_df, features)
    X = np.asarray(metrics_df)
    # Scale data:
    X = loaded_scaler.transform(X)

    matrix = pickle.load(open('{}/reduction_{}.pickle.dat'.format(model_dir, model_name), 'rb'))
    X = matrix.transform(X)

    # Make predictions for given data
    start = time.clock()
    y_pred = loaded_model.predict(X)
    prediction_time = time.clock() - start

    predictions = []
    # Display predictions
    for i, rendition in enumerate(renditions_list):
        predictions.append(y_pred[i])
        if y_pred[i] == -1:
            attack = ''
        else:
            attack = ' not'

        print('{} is{} an attack'.format(rendition, attack))

    if do_profiling:
        print('Features used:', metrics_list)
        print('Total CPU time:', time.clock() - total_start)
        print('Total user time:', time.time() - total_start_user)
        print('Initialization CPU time:', initialize_time)
        print('Initialization user time:', initialize_time_user)
        
        print('Process CPU time:', process_time)
        print('Process user time:', process_time_user)
        print('Prediction CPU time:', prediction_time)
    print(predictions)
    return predictions

def rescale_to_resolution(data, features):
        feat_labels =  ['dimension', 
                        'size',
                        'fps',
                        'temporal_difference-euclidean', 
                        'temporal_difference-manhattan',
                        'temporal_difference-max', 
                        'temporal_difference-mean',
                        'temporal_difference-std', 
                        'temporal_cross_correlation-euclidean', 
                        'temporal_cross_correlation-manhattan',
                        'temporal_cross_correlation-max', 
                        'temporal_cross_correlation-mean',
                        'temporal_cross_correlation-std',
                        'temporal_dct-euclidean', 
                        'temporal_dct-manhattan',
                        'temporal_dct-max', 
                        'temporal_dct-mean',
                        'temporal_dct-std',
                        'temporal_canny-euclidean', 
                        'temporal_canny-manhattan',
                        'temporal_canny-max', 
                        'temporal_canny-mean',
                        'temporal_canny-std',
                        'temporal_gaussian-euclidean', 
                        'temporal_gaussian-manhattan',
                        'temporal_gaussian-max', 
                        'temporal_gaussian-mean',
                        'temporal_gaussian-std',
                        'temporal_gaussian_difference-euclidean', 
                        'temporal_gaussian_difference-manhattan',
                        'temporal_gaussian_difference-max', 
                        'temporal_gaussian_difference-mean',
                        'temporal_gaussian_difference-std',
                        'temporal_gaussian_difference_threshold-euclidean', 
                        'temporal_gaussian_difference_threshold-manhattan',
                        'temporal_gaussian_difference_threshold-max', 
                        'temporal_gaussian_difference_threshold-mean',
                        'temporal_gaussian_difference_threshold-std',
                        'temporal_histogram_distance-euclidean',
                        'temporal_histogram_distance-manhattan',
                        'temporal_histogram_distance-max', 
                        'temporal_histogram_distance-mean',
                        'temporal_histogram_distance-std',
                        'temporal_ssim-euclidean',
                        'temporal_ssim-manhattan',
                        'temporal_ssim-max', 
                        'temporal_ssim-mean',
                        'temporal_ssim-std',
                        'temporal_psnr-euclidean',
                        'temporal_psnr-manhattan',
                        'temporal_psnr-max', 
                        'temporal_psnr-mean',
                        'temporal_psnr-std',
                        'temporal_entropy-euclidean',
                        'temporal_entropy-manhattan',
                        'temporal_entropy-max', 
                        'temporal_entropy-mean',
                        'temporal_entropy-std',
                        'temporal_lbp-euclidean',
                        'temporal_lbp-manhattan',
                        'temporal_lbp-max', 
                        'temporal_lbp-mean',
                        'temporal_lbp-std',
                        'temporal_orb-euclidean',
                        'temporal_orb-manhattan',
                        'temporal_orb-max', 
                        'temporal_orb-mean',
                        'temporal_orb-std',
                        ]
        df = pd.DataFrame(data)
        downscale_features = [
                        'temporal_psnr', 
                        'temporal_ssim', 
                        'temporal_cross_correlation'
                     ]

        upscale_features = [
                            'temporal_difference', 
                            'temporal_dct', 
                            'temporal_canny', 
                            'temporal_gaussian', 
                            'temporal_gaussian_difference', 
                            'temporal_histogram_distance',
                            'temporal_entropy',
                            'temporal_lbp'
                        ]

        for label in feat_labels:

            if label in features:
                
                if label.split('-')[0] in downscale_features:
                    df[label] = df.apply(lambda row: (row[label]/row['dimension']), axis=1)
                    print('Downscaling',label, flush=True)
                elif label.split('-')[0] in upscale_features:
                    df[label] = df.apply(lambda row: (row[label]*row['dimension']), axis=1)
                    print('Upscaling',label, flush=True)
        return df