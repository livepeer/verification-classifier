import click
import pandas as pd
import numpy as np
import time
from scipy.spatial import distance

import sys
sys.path.insert(0, 'scripts/')

from video_asset_processor import video_asset_processor

@click.command()
@click.argument('asset')
@click.option('--renditions', '-rendition', multiple=True)
def cli(asset, renditions):
    click.echo(f"Processing {asset}")
    click.echo(f"Renditions {renditions}")

    start_time = time.time()

    original_asset = asset

    renditions_list = renditions
    metrics_list = ['temporal_canny']
    
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list)
    
    
    asset_metrics_dict = asset_processor.process()
    dict_of_df = {k: pd.DataFrame(v) for k,v in asset_metrics_dict.items()}
    metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
    metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})
    displayed_metric = 'temporal_canny'

    renditions_df = pd.DataFrame()
    frames = []
    for rendition in renditions_list:
        
        rendition_df = metrics_df[metrics_df['path']==rendition][displayed_metric]

        rendition_df = rendition_df.reset_index(drop=True).transpose()
        frames.append(rendition_df)

    renditions_df = pd.concat(frames,axis=1)
    renditions_df.columns = renditions_list
    renditions_df = renditions_df.astype(float)

    x_original = np.array(renditions_df[original_asset].values)

    distances = {}

    for rendition in renditions_list:
        
        x = np.array(renditions_df[rendition].values)
        
        euclidean = distance.euclidean(x_original, x)
        
        distances[rendition] = {'Euclidean': euclidean}

    distances_raw_df = pd.DataFrame.from_dict(distances,orient='index')

    print(distances_raw_df)
     # Collect processing time
    elapsed_time = time.time() - start_time 
    print('Total procesing time: {} s'.format(elapsed_time))
 
if __name__ == '__main__':
    cli()