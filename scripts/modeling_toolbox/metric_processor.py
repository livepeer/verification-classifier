import pandas as pd
import numpy as np
import pywt

class MetricProcessor:

    def __init__(self, features, learning_type, path, reduced=False, bins=0, scale=True):

        self.features = features.copy()
        self.learning_type = learning_type
        self.path = path
        self.reduced = reduced
        self.bins = bins
        self.series_features_list = []
        self.scale = scale
        for feature in features:
            if 'temporal' in feature:
                self.series_features_list.append('{}-series'.format(feature.split('-')[0]))

        self.info_columns = ['attack_ID', 'title', 'attack']

    @staticmethod
    def compute_fft(row, column):
        y = np.fromstring(row[column].replace('[', '').replace(']', ''), 
                                            dtype=np.float, sep=' ')
        n = len(y)
        Y = np.fft.fft(y)/n
        Y = Y[range(int(n/2))]

        return Y

    @staticmethod
    def compute_dwt(row, column, nbins):
        scales = range(1,10)
        waveletname = 'morl'
        signal = np.fromstring(row[column].replace('[', '').replace(']', ''), dtype=np.float, sep=' ')
        coeff, _ = pywt.cwt(signal, scales, waveletname, 1)
        
        coeff_ = coeff[:,:nbins]
        return coeff_[0]

    def tryconvert(self, x):
        try:
            return x.split('/')[-2] 
        except:
            return ''


    def read_and_process_data(self):
        data = pd.read_csv(self.path)
        if self.reduced:
            data = data[:self.reduced]
        renditions_list = ['attack', '1080p', '720p', '480p', '360p', '240p', '144p']

        df = pd.DataFrame(data)
        
        # Fix attack column to contain only its name
        df['attack'] = df['attack'].apply(lambda x: self.tryconvert(x))
 
        if self.scale:
            df = self.rescale_to_resolution(df)

        del data
        attack_IDs = []

        for _, row in df.iterrows():
            if row['attack'] in renditions_list:
                attack_IDs.append(renditions_list.index(row['attack']))
            elif 'watermark' in row['attack']:
                attack_IDs.append(11)
            elif 'low_bitrate_4' in row['attack']:
                attack_IDs.append(12)
            else:
                attack_IDs.append(10)

        if self.bins != 0:
            for column in self.series_features_list:
                print('Computing time series descriptor for ', column)
                for i in range(self.bins):
                    print('Computing bin ', i)
                    # df['{}-hist-{}'.format(column, i)] = df.apply(lambda row: np.histogram((np.fromstring(row[column].replace('[', '').replace(']', ''), 
                    #                         dtype=np.float, sep=' ')), self.bins)[0][i], axis=1)
                    # df['{}-mean-{}'.format(column, i)] = df.apply(lambda row: np.array_split((np.fromstring(row[column].replace('[', '').replace(']', ''), 
                    #                         dtype=np.float, sep=' ')), self.bins)[i].mean(), axis=1)
                    df['{}-dwt-{}'.format(column, i)] = df.apply(lambda row: self.compute_dwt(row, column, self.bins)[i], axis=1)
        
        df['attack_ID'] = attack_IDs

        if self.bins != 0:
            hist_n_means = list(filter(lambda x: '-mean-' in x or '-hist-' in x or '-dwt-' in x, list(df)))
            self.features.extend(hist_n_means)

        df = df.drop(['Unnamed: 0', 'path', 'kind'], axis=1)
        df = df.drop(self.series_features_list, axis=1)        

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
            y_test_all = np.where(df_test_all['attack_ID']>=10, 0, 1)

            x_train = df_train.drop(['title',
                                     'attack',
                                     'attack_ID'], axis=1)

            x_test = df_test.drop(['title',
                                   'attack',
                                   'attack_ID'], axis=1)

            y_train = np.where(df_train['attack_ID']>=10, 0, 1)
            y_test = np.where(df_test['attack_ID']>=10, 0, 1)

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
        feat_labels = [
                       'dimension',
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
                       'temporal_gaussian_difference-euclidean',
                       'temporal_gaussian_difference-manhattan',
                       'temporal_gaussian_difference-max',
                       'temporal_gaussian_difference-mean',
                       'temporal_gaussian_difference-std',
                       'temporal_gaussian_difference_threshold-euclidean',
                       'temporal_gaussian_difference_threshold-manhattan',
                       'temporal_gaussian_difference_threshold-max',
                       'temporal_gaussian_difference_threshold-mean',
                       'temporal_gaussian_difference_threshold-std',
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
                       'temporal_entropy-std',
                       'temporal_lbp-euclidean',
                       'temporal_lbp-manhattan',
                       'temporal_lbp-max',
                       'temporal_lbp-mean',
                       'temporal_lbp-std',
                       'temporal_orb-euclidean',
                       'temporal_orb-manhattan',
                       'temporal_orb-max',
                       'temporal_orb-mean',
                       'temporal_orb-std',
                       'temporal_texture-euclidean',
                       'temporal_texture-manhattan',
                       'temporal_texture-max',
                       'temporal_texture-mean',
                       'temporal_texture-std',
                       'temporal_match-euclidean',
                       'temporal_match-manhattan',
                       'temporal_match-max',
                       'temporal_match-mean',
                       'temporal_match-std',
                      ]
        df = pd.DataFrame(data)
        downscale_features = [
                              'temporal_psnr',
                              'temporal_ssim',
                              'temporal_cross_correlation'
                             ]

        upscale_features = [
                            'temporal_difference', 
                            'temporal_dct', 
                            'temporal_canny', 
                            'temporal_gaussian_mse',
                            'temporal_gaussian_difference', 
                            'temporal_histogram_distance',
                            'temporal_entropy',
                            'temporal_lbp',
                            'temporal_texture',
                            'temporal_match',
                           ]

        for label in feat_labels:

            if label in self.features:
                print(label)
                if label.split('-')[0] in downscale_features:
                    df[label] = df.apply(lambda row: (row[label]/row['dimension']), axis=1)
                elif label.split('-')[0] in upscale_features:
                    df[label] = df.apply(lambda row: (row[label]*row['dimension']), axis=1)
        return df
