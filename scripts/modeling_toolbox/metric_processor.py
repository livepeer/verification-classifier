import pandas as pd
import numpy as np
    
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
    def set_attack_name(x):

        try:
            return x.split('/')[-2]
        except:
            return ''

    @staticmethod
    def set_attack_id(x):

        renditions_list = ['attack', '1080p', '720p', '480p', '360p', '240p', '144p']
        try:
            if x in renditions_list:
                return renditions_list.index(x)
            elif 'watermark' in x:
                return 11
            elif 'low_bitrate_4' in x:
                return 12
            else:
                return 10
        except:
            return ''
        
    def read_and_process_data(self, unique_ID=False):

        data = pd.read_csv(self.path)
        if self.reduced:
            data = data[:self.reduced]

        df = pd.DataFrame(data)
        
        del data

        # Fix attack column to contain only its name
        if unique_ID:
            df['unique_ID'] = df['attack']
            self.info_columns.append('unique_ID')
        df['attack'] = df['attack'].apply(lambda x: self.set_attack_name(x))
        df['attack_ID'] = df['attack'].apply(lambda x: self.set_attack_id(x))

        if self.scale:
            print('Rescaling {}'.format(df.columns))
            df = self.rescale_to_resolution(df)


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

        df = pd.DataFrame(data)
        downscale_features = ['temporal_psnr',
                              'temporal_ssim',
                              'temporal_cross_correlation'
                             ]

        upscale_features = ['temporal_difference',
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

        for label in downscale_features:
            downscale_feature = [feature for feature in self.features if label in feature]
            if downscale_feature:
                for feature in downscale_feature:
                    print('Downscaling', label, feature)
                    df[feature] = df[feature]/df['dimension']

        for label in upscale_features:
            upscale_feature = [feature for feature in self.features if label in feature]
            if upscale_feature:
                for feature in upscale_feature:
                    print('Upscaling', label, feature)
                    df[feature] = df[feature]*df['dimension']

        return df
