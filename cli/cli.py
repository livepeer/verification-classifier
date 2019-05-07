import os
import click
import pickle
import time
import pandas as pd
import numpy as np
import urllib.request

from scipy.spatial import distance

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
    
    metrics_dict = {}
    metrics_list = ['temporal_difference', 'temporal_canny', 'temporal_histogram_distance', 'temporal_cross_correlation', 'temporal_dct']

    original_asset = asset

    renditions_list = list(renditions)

    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 4)

    asset_metrics_dict = asset_processor.process()

    ## Aggregate dictionary with values into a Pandas DataFrame
    dict_of_df = {k: pd.DataFrame(v) for k, v in asset_metrics_dict.items()}
    metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
    metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})

    elapsed_time = time.time() - start_time
    print('Processing time:', elapsed_time)

    renditions_dict = {}
    for rendition in renditions_list:
        rendition_dict = {}
        for metric in metrics_list:

            original_df = metrics_df[metrics_df['path']==original_asset][metric]
            original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)

            rendition_df = metrics_df[metrics_df['path']==rendition][metric]
            rendition_df = rendition_df.reset_index(drop=True).transpose().dropna().astype(float)

            if  'temporal' in metric:
                x_original = np.array(original_df[rendition_df.index].values)
                x_rendition = np.array(rendition_df.values)

                [[manhattan]] = 1/abs(1-distance.cdist(x_original.reshape(1,-1), x_rendition.reshape(1,-1), metric='cityblock'))

                rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original, x_rendition)
                rendition_dict['{}-manhattan'.format(metric)] = manhattan
                rendition_dict['{}-mean'.format(metric)] = np.mean(x_rendition)
                rendition_dict['{}-max'.format(metric)] = np.max(x_rendition)
                rendition_dict['{}-std'.format(metric)] = np.std(x_rendition)
            else:
                rendition_dict[metric] = rendition_df.mean()
            rendition_dict['size'] = os.path.getsize(rendition)
        renditions_dict[rendition] = rendition_dict

    metrics_dict[original_asset] = renditions_dict 

    dict_of_df = {k: pd.DataFrame(v) for k,v in metrics_dict.items()}
    metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
    print(metrics_df)  

    metrics_df['title'] = metrics_df['level_0']
    attack_series = []
    dimensions_series = []
    for _, row in metrics_df.iterrows():
        attack_series.append(row['level_1'].split('/')[-2])

    metrics_df['attack'] = attack_series

    for _, row in metrics_df.iterrows():
        dimension = int(row['attack'].split('_')[0].replace('p',''))
        dimensions_series.append(dimension)

    metrics_df['dimension'] = dimensions_series
    X = metrics_df.drop(['level_0',
                        'title',
                        'attack',
                        'level_1'],
                        axis=1)

    X = np.asarray(X)
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
