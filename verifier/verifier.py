import pickle
import numpy as np
import urllib.request
import time
import json

import sys

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import video_asset_processor


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
    non_numeric_features = ['attack_ID', 'title', 'attack', 'dimension']
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

