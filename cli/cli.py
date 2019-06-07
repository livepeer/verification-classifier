import os
import click
import pickle
import pandas as pd
import numpy as np
import urllib.request
import time

import sys

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import video_asset_processor

@click.command()
@click.argument('asset')
@click.option('--renditions', multiple = True)
@click.option('--model_url')
@click.option('--do_profiling', default=0)
def cli(asset, renditions, model_url, do_profiling):
    # Download model from remote url
    total_start = time.clock()
    
    start = time.clock()
    loaded_model = download_models(model_url)
    download_time = time.clock() - start
    
    
    # Prepare inpuit variables
    original_asset = asset
    renditions_list = list(renditions)
    metrics_list = ['temporal_gaussian', 'temporal_difference', 'temporal_canny', 'temporal_histogram_distance', 'temporal_cross_correlation', 'temporal_dct']

    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 4, do_profiling)
    initialize_time = time.clock() - start
    
    start = time.clock()
    metrics_df = asset_processor.process()
    process_time = time.clock() - start
    
    # Cleanup the resulting pandas dataframe and convert it to a numpy array
    # to pass to the prediction model
    for column in metrics_df.columns:
        if 'series' in column:
            metrics_df =  metrics_df.drop([column], axis=1)
    metrics_df =  metrics_df.drop('title', axis=1)
    metrics_df =  metrics_df.drop('path', axis=1)
    metrics_df = metrics_df.drop('attack', axis=1)
    metrics_df = metrics_df[['dimension', 'fps', 'size', 'temporal_canny-euclidean',
       'temporal_canny-manhattan', 'temporal_canny-max', 'temporal_canny-mean',
       'temporal_canny-std', 'temporal_cross_correlation-euclidean',
       'temporal_cross_correlation-manhattan',
       'temporal_cross_correlation-max', 'temporal_cross_correlation-mean',
       'temporal_cross_correlation-std', 'temporal_dct-euclidean',
       'temporal_dct-manhattan', 'temporal_dct-max', 'temporal_dct-mean',
       'temporal_dct-std', 'temporal_difference-euclidean',
       'temporal_difference-manhattan', 'temporal_difference-max',
       'temporal_difference-mean', 'temporal_difference-std',
       'temporal_gaussian-euclidean', 'temporal_gaussian-manhattan',
       'temporal_gaussian-max', 'temporal_gaussian-mean',
       'temporal_gaussian-std', 'temporal_histogram_distance-euclidean',
       'temporal_histogram_distance-manhattan',
       'temporal_histogram_distance-max', 'temporal_histogram_distance-mean',
       'temporal_histogram_distance-std']
                             ]
    X = np.asarray(metrics_df)

    # Make predictions for given data
    start = time.clock()
    y_pred = loaded_model.predict(X)
    prediction_time = time.clock() - start
    

    # Display predictions
    i = 0
    for rendition in renditions_list:
        if y_pred[i] == 0:
            attack = ''
        else:
            attack = 'not'

        print('{} is {} an attack'.format(rendition, attack))
        i = i + 1
    
    
    if do_profiling:
        print ('Total time:', time.clock() - total_start)
        print ('Download time:', download_time)
        print ('Initialization time:', initialize_time)
        print ('Process time:', process_time)
        print ('Prediction time:', prediction_time)


def download_models(url):

    print ("Model download started!")
    filename, _ = urllib.request.urlretrieve(url, filename=url.split('/')[-1])

    print("Model downladed")
    
    return pickle.load(open(filename, "rb"))

if __name__ == '__main__':
    cli()
