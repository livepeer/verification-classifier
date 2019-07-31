import click
import pickle
import numpy as np
import urllib.request
import time
import tarfile
import json

import sys

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import video_asset_processor


@click.command()
@click.argument('asset')
@click.option('--renditions', multiple=True)
@click.option('--max_samples', type=int, default=10)
@click.option('--do_profiling', default=0)
def cli(asset, renditions, do_profiling, max_samples):
    seconds = 2
    
    # Download model from remote url
    total_start = time.clock()
    total_start_user = time.time()

    model_url = 'https://storage.googleapis.com/verification-models/verification.tar.gz'
    model_name = 'OCSVM'
    scaler_type = 'StandardScaler'
    learning_type = 'UL'
    start = time.clock()
    download_models(model_url)
    download_time = time.clock() - start
    loaded_model = pickle.load(open('{}.pickle.dat'.format(model_name), 'rb'))
    loaded_scaler = pickle.load(open('{}_{}.pickle.dat'.format(learning_type, scaler_type), 'rb'))

    with open('param_{}.json'.format(model_name)) as json_file:
        params = json.load(json_file)
        features = params['features']

    # Prepare input variables
    original_asset = asset
    renditions_list = list(renditions)
    metrics_list = ['temporal_gaussian', 'temporal_dct', 'temporal_orb']
    
    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    start_user = time.time()
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, seconds, max_samples, do_profiling)
    initialize_time = time.clock() - start
    initialize_time_user = time.time() - start_user

    start = time.clock()
    start_user = time.time()
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

    matrix = pickle.load(open('reduction_{}.pickle.dat'.format(model_name), 'rb'))
    X = matrix.transform(X)

    # Make predictions for given data
    start = time.clock()
    y_pred = loaded_model.predict(X)
    prediction_time = time.clock() - start

    # Display predictions
    for i, rendition in enumerate(renditions_list):
        if y_pred[i] == -1:
            attack = ''
        else:
            attack = ' not'

        print('{} is{} an attack'.format(rendition, attack))

    if do_profiling:
        print('Features used:', metrics_list)
        print('Total CPU time:', time.clock() - total_start)
        print('Total user time:', time.time() - total_start_user)
        print('Download time:', download_time)
        print('Initialization CPU time:', initialize_time)
        print('Initialization user time:', initialize_time_user)
        
        print('Process CPU time:', process_time)
        print('Process user time:', process_time_user)
        print('Prediction CPU time:', prediction_time)
        


def download_models(url):

    print('Model download started!')
    filename, _ = urllib.request.urlretrieve(url, filename=url.split('/')[-1])

    print('Model downloaded')

    with tarfile.open(filename) as tf:
        tf.extractall()


if __name__ == '__main__':
    cli()
