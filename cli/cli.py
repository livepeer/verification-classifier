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
@click.option('--do_profiling', default=0)
def cli(asset, renditions, do_profiling):
    # Download model from remote url
    total_start = time.clock()
    model_url = 'https://github.com/livepeer/verification-classifier/blob/master/machine_learning/output/models/' \
                'model.tar.gz?raw=true'
    model_name = 'XGBoost'
    scaler_type = 'MinMaxScaler'
    learning_type = 'SL'
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
    metrics_list = ['temporal_gaussian']

    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 1, do_profiling)
    initialize_time = time.clock() - start

    start = time.clock()
    metrics_df = asset_processor.process()
    process_time = time.clock() - start

    prediction_time = 0
    try:
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
        i = 0
        for rendition in renditions_list:
            if y_pred[i] == 0:
                attack = ''
            else:
                attack = ' not'

            print('{} is{} an attack'.format(rendition, attack))
            i = i + 1
    except:
        print('Prediction failed')

    print('Total time:', time.clock() - total_start)
    print('Download time:', download_time)
    print('Initialization time:', initialize_time)
    print('Process time:', process_time)
    print('Prediction time:', prediction_time)


def download_models(url):

    print('Model download started!')
    filename, _ = urllib.request.urlretrieve(url, filename=url.split('/')[-1])

    print('Model downloaded')

    with tarfile.open(filename) as tf:
        tf.extractall()


if __name__ == '__main__':
    cli()
