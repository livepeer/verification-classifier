import sys
import time
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from video_asset_processor import video_asset_processor


def read_metric_log(path, metric):
    if metric == 'vmaf':
        with open(path) as f:
            for line in f:
                if '= ' in line:
                    return float(line.split('= ')[-1])
    if metric == 'ms-ssim':
        ms_ssim_df = pd.read_csv(path)
        return ms_ssim_df['ms-ssim'].mean()

if __name__ == "__main__":

    metrics_list = ['temporal_difference', 'temporal_canny']

    renditions_folders = [
        '1080p',
        #'1080p_watermark',
        #'1080p_flip_vertical',
        #'1080p_rotate_90_clockwise',
        '720p',
        #'720p_watermark',
        #'720p_flip_vertical',
        #'720p_rotate_90_clockwise',
        '480p',
        #'480p_watermark',
        #'480p_flip_vertical',
        #'480p_rotate_90_clockwise',
        '360p',
        #'360p_watermark',
        #'360p_flip_vertical',
        #'360p_rotate_90_clockwise',
        '240p',
        #'240p_watermark',
        #'240p_flip_vertical',
        #'240p_rotate_90_clockwise',
        '144p',
        #'144p_watermark',
        #'144p_flip_vertical',
        #'144p_rotate_90_clockwise',
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

    dict_of_df = {k: pd.DataFrame(v) for k, v in metrics_dict.items()}
    metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
    metrics_df = pd.read_csv('../output/metrics.csv')
    metrics_df = metrics_df.drop(['Unnamed: 0'], axis=1)
    metrics_path = '../output'
    real_path = os.path.realpath(metrics_path)
    extra_metrics = ['vmaf', 'ms-ssim']

    for index, row in metrics_df.iterrows():
        for metric in extra_metrics:

            asset_name = row['level_0'].split('/')[-1].split('.')[0]
            attack = row['level_1'].split('/')[2]
            dimension = row['level_1'].split('/')[2].split('_')[0].replace('p', '')
            attack_name = attack.replace('{}p'.format(dimension), dimension)
            log_path = '{}/{}/{}/{}/{}_{}.log'.format(metrics_path, metric, attack_name, asset_name, asset_name,
                                                      dimension)

            print('LEVEL 0', row['level_0'])
            print('LEVEL 1:', row['level_1'])
            print('ASSET NAME:', asset_name)
            print('ATTACK:', attack)
            print('DIMENSION', dimension)
            print('ATTACK NAME', attack_name)
            print('PATH:', log_path)

            if os.path.isfile(log_path):
                print('ADDING:', log_path)
                print('*****************************')
                metric_value = read_metric_log(log_path, metric)
                metrics_df.at[index, metric] = metric_value

    metrics_df.to_csv('../output/metrics.csv')
    metrics_df.head()
