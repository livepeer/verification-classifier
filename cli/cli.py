from video_asset_processor import video_asset_processor
import click
import pandas as pd
import numpy as np
from scipy.spatial import distance

import sys

sys.path.insert(0, 'scripts/')


@click.command()
@click.argument('asset')
@click.option('-r', '--renditions', multiple=True)
@click.option('-m', '--metrics', multiple=True)
def cli(asset, renditions, metrics):
    implemented_metrics = {
        'temporal_canny',
        'temporal_difference',
        'temporal_psnr',
        'temporal_mse',
        'histogram_distance',
        'hash_euclidean',
        'hash_hamming',
        'hash_cosine'
    }

    for metric in metrics:
        if metric not in implemented_metrics:
            raise click.BadOptionUsage(metric,
                                       '{} is not a valid metric, try with: {}'.format(metric, implemented_metrics))

    original_asset = asset

    renditions_list = [original_asset] + list(renditions)

    asset_processor = video_asset_processor(original_asset, renditions_list, metrics)

    asset_metrics_dict = asset_processor.process()
    dict_of_df = {k: pd.DataFrame(v) for k, v in asset_metrics_dict.items()}
    metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
    metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})

    distances_result = {}
    for displayed_metric in metrics:
        frames = []
        for rendition in renditions_list:
            rendition_df = metrics_df[metrics_df['path'] == rendition][displayed_metric]

            rendition_df = rendition_df.reset_index(drop=True).transpose()
            frames.append(rendition_df)

        renditions_df = pd.concat(frames, axis=1)
        renditions_df.columns = renditions_list
        renditions_df = renditions_df.astype(float)

        x_original = np.array(renditions_df[original_asset].values)

        distances = {}

        for rendition in renditions_list:
            x = np.array(renditions_df[rendition].values)

            euclidean = distance.euclidean(x_original, x)

            distances[rendition] = {'Euclidean': euclidean}

        distances_result[displayed_metric] = pd.DataFrame.from_dict(distances, orient='index').to_json(orient="index")

    print(distances_result)


if __name__ == '__main__':
    cli()
