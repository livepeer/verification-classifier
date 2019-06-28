import click
import numpy as np
import urllib.request
import time
import tarfile
import sys

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import video_asset_processor


@click.command()
@click.argument('asset')
@click.option('--renditions', multiple=True)
@click.option('--do_profiling', default=0)
def cli(asset, renditions, do_profiling, seconds=1):
    total_start = time.clock()

    # Model:
    a = -0.002317248118922172
    b = 0.07988856632766324

    features = 'temporal_gaussian-mean'

    # Prepare input variables
    original_asset = asset
    renditions_list = list(renditions)
    metrics_list = ['temporal_gaussian']

    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, seconds, do_profiling)
    initialize_time = time.clock() - start

    start = time.clock()
    metrics_df = asset_processor.process()
    process_time = time.clock() - start

    # Clean dataframe and compute numpy array of features
    dimensions = np.asarray(metrics_df['dimension'])
    metrics_df = metrics_df[features]
    X = np.asarray(metrics_df)

    # Make predictions for given data
    start = time.clock()
    y_pred = X > 10 ** (a*dimensions + b)
    prediction_time = time.clock() - start

    # Display predictions
    for i, rendition in enumerate(renditions_list):
        if y_pred[i]:
            attack = ''
        else:
            attack = ' not'

        print('{} is{} an attack'.format(rendition, attack))

    if do_profiling:
        print('Total time:', time.clock() - total_start)

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
