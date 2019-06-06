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
                                     'temporal_histogram_distance-series']

    def read_and_process_data(self):
        data = pd.read_csv(self.path)
        if self.reduced:
            data = data[:20000]

        df = pd.DataFrame(data)
        del data
        histogram_range = np.arange(self.bins)
        attack_IDs = []

        for column in df.columns:
            if 'series' in column:
                for i in histogram_range:
                    df['{}-hist-{}'.format(column, i)] = 0.0
                    df['{}-mean-{}'.format(column, i)] = 0.0

        for row_index, row in df.iterrows():

            if row['attack'] in ['1080p', '720p', '480p', '360p', '240p', '144p']:
                attack_IDs.append(1)
            else:
                attack_IDs.append(0)

            if self.bins != 0:
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

        df = df[self.features]
        df = df.dropna()

        return df





