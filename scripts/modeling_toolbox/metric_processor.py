import pandas as pd
import numpy as np


class MetricProcessor:

    def __init__(self, features, learning_type, path, reduced=False, bins=0):
        self.features = features
        self.learning_type = learning_type
        self.path = path
        self.reduced = reduced
        self.bins = bins
        self.series_features_list = ['temporal_canny-series',
                                     'temporal_cross_correlation-series',
                                     'temporal_dct-series',
                                     'temporal_difference-series',
                                     'temporal_histogram_distance-series',
                                     'temporal_gaussian-series']
        self.info_columns = ['attack_ID', 'title', 'attack']

    def read_and_process_data(self):
        data = pd.read_csv(self.path)
        if self.reduced:
            data = data[:20000]
        renditions_list = ['attack', '720p', '480p', '360p', '240p', '144p']

        df = pd.DataFrame(data)

        df = self.rescale_to_resolution(df)
        del data
        attack_IDs = []

        for row_index, row in df.iterrows():

            if row['attack'] in renditions_list:
                attack_IDs.append(renditions_list.index(row['attack']))
            elif 'watermark' in row['attack']:
                    attack_IDs.append(11)
            elif 'low_bitrate_4' in row['attack']:
                    attack_IDs.append(12)
            else:
                attack_IDs.append(10)

            if self.bins != 0:
                histogram_range = np.arange(self.bins)
                for column in df.columns:
                    if 'series' in column:
                        for i in histogram_range:
                            df['{}-hist-{}'.format(column, i)] = 0.0
                            df['{}-mean-{}'.format(column, i)] = 0.0

                for column in self.series_features_list:
                    time_series = np.fromstring(row[column].replace('[', '').replace(']', ''), dtype=np.float, sep=' ')
                    histogram = np.histogram(time_series, bins=histogram_range, density=True)[0]
                    mean_series = np.array_split(time_series, self.bins)

                    for i in np.arange(len(histogram)):
                        df.at[row_index, '{}-hist-{}'.format(column, i)] = histogram[i]
                        df.at[row_index, '{}-mean-{}'.format(column, i)] = mean_series[i].mean()

        df['attack_ID'] = attack_IDs

        if self.bins != 0:
            hist_n_means = list(filter(lambda x: '-mean-' in x or '-hist-' in x, list(df)))
            self.features.extend(hist_n_means)

        df = df.drop(['Unnamed: 0', 'path', 'kind'], axis=1)
        df = df.drop(self.series_features_list, axis=1)
        df = df.dropna(axis=0)

        columns = self.features
        columns.extend(self.info_columns)
        df = df[columns]
        df = df.dropna()

        return df

    def split_test_and_train(self, df, train_prop=0.8):
        if self.learning_type == 'SL':

            num_train = int(df.shape[0]*train_prop)

            df_train_all = df[0:num_train]
            df_test_all = df[num_train:]

            df_train_1 = df_train_all[df_train_all['attack_ID'] < 10]
            df_train_0 = df_train_all[df_train_all['attack_ID'] >= 10]

            df_sample_train = df_train_0.sample(df_train_1.shape[0])
            df_train = df_train_1.append(df_sample_train)
            df_train = df_train.sample(frac=1)

            x_test_all = df_test_all.drop(['title',
                                           'attack',
                                           'attack_ID'], axis=1)
            df_test_1 = df_test_all[df_test_all['attack_ID'] < 10]
            df_test_0 = df_test_all[df_test_all['attack_ID'] >= 10]

            df_sample_test = df_test_0.sample(df_test_1.shape[0])
            df_test = df_test_1.append(df_sample_test)
            df_test = df_test.sample(frac=1)

            x_test_all = np.asarray(x_test_all)
            y_test_all = df_test_all['attack_ID']

            x_train = df_train.drop(['title',
                                     'attack',
                                     'attack_ID'], axis=1)

            x_test = df_test.drop(['title',
                                   'attack',
                                   'attack_ID'], axis=1)

            y_train = df_train['attack_ID']
            y_test = df_test['attack_ID']

            return (x_test_all, y_test_all), (x_train, y_train), (x_test, y_test)

        elif self.learning_type == 'UL':

            df_1 = df[df['attack_ID'] < 10]
            df_0 = df[df['attack_ID'] >= 10]

            num_train = int(df_1.shape[0]*train_prop)
            df_train = df_1[0:num_train]
            df_test = df_1[num_train:]
            df_attacks = df_0

            df_train = df_train.sample(frac=1)
            df_test = df_test.sample(frac=1)
            df_attacks = df_attacks.sample(frac=1)

            x_train = df_train.drop(['title',
                                     'attack',
                                     'attack_ID'], axis=1)
            x_train = np.asarray(x_train)

            x_test = df_test.drop(['title',
                                   'attack',
                                   'attack_ID'], axis=1)
            x_test = np.asarray(x_test)

            x_attacks = df_attacks.drop(['title',
                                         'attack',
                                         'attack_ID'], axis=1)
            x_attacks = np.asarray(x_attacks)

            return (x_train, x_test, x_attacks), (df_train, df_test, df_attacks)
        else:
            print('Unknown learning type. Use UL for unsupervised learning and SL for supervised learning')
    def rescale_to_resolution(self, data):
        feat_labels =  ['dimension', 
                        'size',
                        'fps',
                        'temporal_difference-euclidean', 
                        'temporal_difference-manhattan',
                        'temporal_difference-max', 
                        'temporal_difference-mean',
                        'temporal_difference-std', 
                        'temporal_cross_correlation-euclidean', 
                        'temporal_cross_correlation-manhattan',
                        'temporal_cross_correlation-max', 
                        'temporal_cross_correlation-mean',
                        'temporal_cross_correlation-std',
                        'temporal_dct-euclidean', 
                        'temporal_dct-manhattan',
                        'temporal_dct-max', 
                        'temporal_dct-mean',
                        'temporal_dct-std',
                        'temporal_canny-euclidean', 
                        'temporal_canny-manhattan',
                        'temporal_canny-max', 
                        'temporal_canny-mean',
                        'temporal_canny-std',
                        'temporal_gaussian-euclidean', 
                        'temporal_gaussian-manhattan',
                        'temporal_gaussian-max', 
                        'temporal_gaussian-mean',
                        'temporal_gaussian-std',
                        'temporal_histogram_distance-euclidean',
                        'temporal_histogram_distance-manhattan',
                        'temporal_histogram_distance-max', 
                        'temporal_histogram_distance-mean',
                        'temporal_histogram_distance-std',
                        'temporal_ssim-euclidean',
                        'temporal_ssim-manhattan',
                        'temporal_ssim-max', 
                        'temporal_ssim-mean',
                        'temporal_ssim-std',
                        'temporal_psnr-euclidean',
                        'temporal_psnr-manhattan',
                        'temporal_psnr-max', 
                        'temporal_psnr-mean',
                        'temporal_psnr-std',
                        'temporal_entropy-euclidean',
                        'temporal_entropy-manhattan',
                        'temporal_entropy-max', 
                        'temporal_entropy-mean',
                        'temporal_entropy-std'
                        ]
        df = pd.DataFrame(data)
        downscale_features = ['temporal_psnr', 
                      'temporal_ssim', 
                      'temporal_cross_correlation'
                     ]

        upscale_features = ['temporal_difference', 
                            'temporal_dct', 
                            'temporal_canny', 
                            'temporal_gaussian', 
                            'temporal_histogram_distance',
                            'temporal_entropy'
                        ]

        for label in feat_labels:
            if label.split('-')[0] in downscale_features:
                df[label] = df.apply(lambda row: (row[label]/row['dimension']), axis=1)
            elif label.split('-')[0] in upscale_features:
                df[label] = df.apply(lambda row: (row[label]*row['dimension']), axis=1)
        return df
