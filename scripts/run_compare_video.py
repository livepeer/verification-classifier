import sys
import time
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from video_asset_processor import video_asset_processor

if __name__ == "__main__":

    metrics_list = ['temporal_difference', 'temporal_canny']

    renditions_folders = [
        '1080p',
        '720p',
        '480p',
        '360p',
        '240p',
        '144p'
    ]
    originals_path = '../data/{}/'

    metrics_dict = {}
    list = os.listdir(originals_path.format('1080p'))  # dir is your directory path
    number_assets = len(list)
    print('Number of assets: {}'.format(number_assets))
    count = 0
    total_time = 0
    for original_asset in glob.iglob(originals_path.format('1080p') + '**', recursive=False):
        count += 1
        if os.path.isfile(original_asset):  # filter dirs
            print('Processing asset {} of {}: {}'.format(count, number_assets, original_asset))
            start_time = time.time()
            renditions_list = []

            for folder in renditions_folders:
                rendition_folder = originals_path.format(folder)
                renditions_list.append(rendition_folder + os.path.basename(original_asset))

            asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list)
            asset_metrics_dict = asset_processor.process()

            dict_of_df = {k: pd.DataFrame(v) for k, v in asset_metrics_dict.items()}

            metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
            metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})

            renditions_dict = {}
            for rendition in renditions_list:
                rendition_dict = {}
                for metric in metrics_list:

                    original_df = metrics_df[metrics_df['path'] == original_asset][metric]
                    original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)

                    rendition_df = metrics_df[metrics_df['path'] == rendition][metric]
                    rendition_df = rendition_df.reset_index(drop=True).transpose().dropna().astype(float)

                    if 'temporal' in metric:
                        x_original = np.array(original_df[rendition_df.index].values)
                        x_rendition = np.array(rendition_df.values)

                        rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original, x_rendition)
                        rendition_dict['{}-cosine'.format(metric)] = distance.cosine(x_original, x_rendition)

                    else:
                        rendition_dict[metric] = rendition_df.mean()

                renditions_dict[rendition] = rendition_dict

            metrics_dict[original_asset] = renditions_dict

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            print('Elapsed time:', elapsed_time)
    print('Total time:', total_time)