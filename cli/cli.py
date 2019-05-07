import os
import click
import pickle
import time
import pandas as pd
import numpy as np
import urllib.request

import sys

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import video_asset_processor

@click.command()
@click.argument('asset')
@click.option('-r', '--renditions', multiple=True)
@click.option('-r', '--model_url')
def cli(asset, renditions, model_url):
    start_time = time.time()

    loaded_model = download_models(model_url)
    
    original_asset = asset

    renditions_list = list(renditions)
    metrics_list = ['temporal_difference', 'temporal_canny', 'temporal_histogram_distance', 'temporal_cross_correlation', 'temporal_dct']

    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 4)

    metrics_df = asset_processor.process()


    X = np.asarray(metrics_df)
    # make predictions for given data
    y_pred = loaded_model.predict(X)

    print(y_pred)
    elapsed_time = time.time() - start_time
    print('Prediction time:', elapsed_time)

def download_models(url):

    print ("Model download started!")
    filename, _ = urllib.request.urlretrieve(url, filename=url.split('/')[-1])

    print("Model downladed")
    
    return pickle.load(open(filename, "rb"))

if __name__ == '__main__':
    cli()
