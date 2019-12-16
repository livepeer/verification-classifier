"""
Module to generate an interactive app to visualize and train a QoE predictive model
using BRISQUE as input
It relies of Streamlite library for the visualization and display of widgets
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import StandardScaler

st.title('QoE model predictor')

DATA_URI_BRISQUE = '../../cloud_functions/data-brisque-large.csv'
DATA_URI_QOE = '../../cloud_functions/data-qoe-metrics-large.csv'

def compute_brisque_features_aggregators(row):
    features = []
    for column in row.index:
        if row[column].shape[0] == 36:
            features.append(row[column])
    row['mean'] = np.mean(features, axis=0)
    row['std'] = np.std(features, axis=0)
    row['mean_dx'] = np.mean(np.diff(features, axis=0), axis=0)
    row['std_dx'] = np.std(np.diff(features, axis=0), axis=0)
    try:
        row['max_dx'] = np.max(np.diff(features, axis=0), axis=0)
    except:
        row['max_dx'] = 0
    return row

@st.cache
def load_data(data_uri, nrows):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame suitable for model training.
    nrows limits the amount of data displayed for optimization
    """
    data_df = pd.read_csv(data_uri, nrows=nrows)

    lowercase = lambda x: str(x).lower()
    data_df.rename(lowercase, axis='columns', inplace=True)
    if 'unnamed: 0' in data_df.columns:
        data_df.drop('unnamed: 0', axis='columns', inplace=True)
    if 'kind' in data_df.columns:
        data_df.drop('kind', axis='columns', inplace=True)

    if 'qoe' in data_uri:
        data_df.rename(columns={'attack':'rendition', 'title':'source'}, inplace=True)
        data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
        data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.split('_')[0]))
        data_df['crf'] = data_df['rendition'].apply(lambda x: x.split('_')[-1])
        data_df['source'] = data_df['source'].apply(lambda x: x.split('/')[-1])
    else:
        names = data_df['source']
        data_df = data_df.drop('source', axis=1)
        sorted_columns = str(sorted([int(i) for i in data_df.columns])).replace('[', '').replace(']', '').split(', ')
        data_df = data_df.reindex(columns=sorted_columns)
        for column in data_df.columns:
                data_df[column] = data_df[column].astype(str).apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''),
                                                                                            dtype=np.float, sep=' '))
        data_df = data_df.apply(lambda row: compute_brisque_features_aggregators(row), axis=1)
        data_df['source'] = names
    return data_df

def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """

    return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]

def plot_rd_curves(df_qoe):
    """
    Display difference between predicted ssim and measured ssim
    according to their tamper classification
    """

    metrics = list(df_qoe.columns)
    asset = st.selectbox('Which asset to represent?', list(df_qoe['source'].unique()))
    metric_x = st.selectbox('Which metric to represent for X?', metrics, index=metrics.index('crf'))
    metric_y = st.selectbox('Which metric to represent for Y?', metrics, index=metrics.index('temporal_ssim-mean'))

    rate_distortion_df = df_qoe[[metric_x, metric_y, 'pixels', 'dimension_y', 'rendition']][df_qoe['source'] == asset]

    data = []
    dimensions = rate_distortion_df['dimension_y'].unique()
    dimensions.sort()
    for dimension in dimensions:

        trace = go.Scatter(x=rate_distortion_df[rate_distortion_df['dimension_y'] == dimension][metric_x],
                           y=rate_distortion_df[rate_distortion_df['dimension_y'] == dimension][metric_y],
                           mode='markers',
                           marker=dict(color=dimension,
                                       opacity=0.8,
                                       line=dict(width=0)
                                       ),
                           hovertext=rate_distortion_df['rendition'],
                           name=str(dimension),
                           )

        data.append(trace)
   
    fig = go.Figure(data=data)
    fig.update_layout(title="{} vs {}".format(metric_x, metric_y),
                      yaxis_title=metric_y,
                      xaxis_title=metric_x
                    )
    fig.update_layout(legend=go.layout.Legend(x=0,
                            y=1,
                            traceorder="normal",
                            font=dict(family="sans-serif",
                                    size=12,
                                    color="black"
                                    ),
                            bgcolor="LightSteelBlue",
                            bordercolor="Black",
                            borderwidth=2
                        )
    )
    st.plotly_chart(fig)

def train_features(models_dict, pools_dict, features, x_train, x_test, y_train, y_test):
    """
    Function to aggregate models from a set of features
    """
    learn_mape_train_df = pd.DataFrame()
    learn_rmse_train_df = pd.DataFrame()
    learn_mape_test_df = pd.DataFrame()
    learn_rmse_test_df = pd.DataFrame()
    categorical_features_indices = []

    for feature in features:

        y_train = pd.DataFrame(data=y_train, columns=features)
        y_test = pd.DataFrame(data=y_test, columns=features)

        train_pool = Pool(data=x_train,
                          label=y_train[feature],
                          cat_features=categorical_features_indices)

        num_trees = 500
        loss_funct = 'MAPE'
        depth = 1
        if 'ssim' in feature:
            loss_funct = 'MAE'
            depth = 10
            num_trees = 200

        models_dict[feature] = CatBoostRegressor(depth=depth,
                                                 num_trees=num_trees,
                                                 l2_leaf_reg=0.2,
                                                 learning_rate=0.05,
                                                 loss_function=loss_funct
                                                )
        #train the model
        print('Training QoE model:', feature)
        models_dict[feature].fit(train_pool)

        pools_dict[feature] = Pool(data=x_test,
                                   label=y_test[feature],
                                   cat_features=categorical_features_indices)

        learn_mape_train_df[feature] = models_dict[feature].eval_metrics(train_pool, ['MAPE'])['MAPE']
        learn_mape_test_df[feature] = models_dict[feature].eval_metrics(pools_dict[feature], ['MAPE'])['MAPE']

        learn_rmse_train_df[feature] = models_dict[feature].eval_metrics(train_pool, ['RMSE'])['RMSE']
        learn_rmse_test_df[feature] = models_dict[feature].eval_metrics(pools_dict[feature], ['RMSE'])['RMSE']

    st.write('QoE model test set MAPE:')
    st.write(learn_mape_test_df.min())
    st.write(learn_mape_test_df.min().describe())

    return models_dict, pools_dict
@st.cache
def predict_qoe(data_df):
    """
    Function to train model from given dataset
    """
    num_train = int(data_df.shape[0] * 0.8)

    train_data = data_df.sample(num_train)
    st.write('Train:', train_data.shape)
    test_data = data_df[~data_df.index.isin(train_data.index)]
    st.write('Test:', test_data.shape)
    x_features = [feature for feature in data_df.columns if 'input' in feature]
    x_features.append('dimension_y')
    x_features.append('pixels')
    x_features.append('crf')
    x_features.append('288_45_ssim')
    x_features.append('288_45_size')
    x_features.append('288_45_pixels')
    
    ssim_features = [feature for feature in data_df.columns if 'ssim' in feature]
    bitrate_features = [feature for feature in data_df.columns if 'size' in feature]
    
    ssim_features.remove('288_45_ssim')
    ssim_features.sort()
    bitrate_features.remove('288_45_size')
    bitrate_features.sort()

    st.write(train_data[x_features].head(), 'Train')
    st.write(train_data[ssim_features].head(), 'Test SSIM')
    st.write(train_data[bitrate_features].head(), 'Test Size')

    x_train = np.asarray(train_data[x_features])
    x_test = np.asarray(test_data[x_features])
    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    ssim_scaler = StandardScaler()
    bitrate_scaler = StandardScaler()
    ssim_train = ssim_scaler.fit_transform(train_data[ssim_features].values)
    bitrate_train = bitrate_scaler.fit_transform(train_data[bitrate_features].values)
    ssim_test = ssim_scaler.transform(test_data[ssim_features].values)
    bitrate_test = bitrate_scaler.transform(test_data[bitrate_features].values)

    models_dict = dict()
    pools_dict = dict()

    x_train = pd.DataFrame(x_train, columns=x_features, index=train_data.index)
    x_test = pd.DataFrame(x_test, columns=x_features, index=test_data.index)
    st.write('XTEST SHAPE:', x_test.shape)
    models_dict, pools_dict = train_features(models_dict,
                                            pools_dict,
                                            ssim_features,
                                            x_train,
                                            x_test,
                                            ssim_train,
                                            ssim_test)

    models_dict, pools_dict = train_features(models_dict,
                                            pools_dict,
                                            bitrate_features,
                                            x_train,
                                            x_test,
                                            bitrate_train,
                                            bitrate_test)

    pred_plot_ssim_df = pd.DataFrame(index=test_data.index)
    pred_plot_bitrate_df = pd.DataFrame(index=test_data.index)
    true_plot_ssim_df = pd.DataFrame(index=test_data.index)
    true_plot_bitrate_df = pd.DataFrame(index=test_data.index)


    for feature_ssim in ssim_features:
        pred_plot_ssim_df[feature_ssim] = models_dict[feature_ssim].predict(pools_dict[feature_ssim])
        true_plot_ssim_df[feature_ssim] = test_data[feature_ssim]
    for feature_bitrate in bitrate_features:
        pred_plot_bitrate_df[feature_bitrate] = models_dict[feature_bitrate].predict(pools_dict[feature_bitrate])
        true_plot_bitrate_df[feature_bitrate] = test_data[feature_bitrate]

    pred_plot_ssim_df = pd.DataFrame(data=ssim_scaler.inverse_transform(pred_plot_ssim_df.values),
                                     columns=ssim_features,
                                     index=test_data.index)
    pred_plot_bitrate_df = pd.DataFrame(data=bitrate_scaler.inverse_transform(pred_plot_bitrate_df.values),
                                        columns=bitrate_features,
                                        index=test_data.index)
    
    pred_plot_ssim_df['source'] = test_data['source']
    pred_plot_bitrate_df['source'] = test_data['source']

    st.write(pred_plot_ssim_df, 'SSIM Predictions')
    st.write(pred_plot_bitrate_df, 'Bitrate Predictions')

    return pred_plot_ssim_df, pred_plot_bitrate_df, test_data[ssim_features], test_data[bitrate_features]

def aggregate_renditions(row, df_renditions):
    """
    Function to assign outputs to inputs from a dissaggregated dataframe
    containing different renditions
    """

    metric = 'temporal_ssim-mean'

    asset = row['source']
    rate_distortion_df = df_renditions[[metric, 'size', 'pixels', 'rendition']][df_renditions['source'] == asset]

    for rendition in rate_distortion_df['rendition']:
        row['{}_size'.format(rendition)] = rate_distortion_df[rate_distortion_df['rendition'] == rendition]['size'].values[0]
        row['{}_ssim'.format(rendition)] = rate_distortion_df[rate_distortion_df['rendition'] == rendition][metric].values[0]
        row['{}_pixels'.format(rendition)] = rate_distortion_df[rate_distortion_df['rendition'] == rendition]['pixels'].values[0]

    return row

def setup_train(df_qoe, df_brisque):
    """
    Function to combine inputs and outputs from several daaframes into one
    """

    df_train = pd.DataFrame()

    df_train['source'] = df_brisque['source']
    df_train['input_std'] = df_brisque['std'].values
    df_train['input_mean'] = df_brisque['mean'].values
    df_train['input_mean_dx'] = df_brisque['mean_dx'].values
    df_train['input_std_dx'] = df_brisque['std_dx'].values
    # df_train['input_max_dx'] = df_brisque['max_dx'].values

    df_train['dimension_y'] = df_qoe['dimension_y']
    df_train['pixels'] = df_qoe['pixels']
    df_train['crf'] = df_qoe['crf']

    df_train = df_train.dropna(axis='rows')
    for brisque_feature in range(36):
        input_feature_label = 'input_mean_{}'.format(brisque_feature)
        df_train[input_feature_label] = df_train['input_mean'].apply(lambda x: x[brisque_feature])
        input_feature_label = 'input_std_{}'.format(brisque_feature)
        df_train[input_feature_label] = df_train['input_std'].apply(lambda x: x[brisque_feature])

        input_feature_label = 'input_mean_dx_{}'.format(brisque_feature)
        df_train[input_feature_label] = df_train['input_mean_dx'].apply(lambda x: x[brisque_feature])
        input_feature_label = 'input_std_dx_{}'.format(brisque_feature)
        df_train[input_feature_label] = df_train['input_std_dx'].apply(lambda x: x[brisque_feature])
        # input_feature_label = 'input_max_dx_{}'.format(brisque_feature)
        # df_train[input_feature_label] = df_train['input_max_dx'].apply(lambda x: x[brisque_feature])

    df_train = df_train.apply(lambda row: aggregate_renditions(row, df_qoe), axis=1)
    df_train = df_train.drop(['input_mean',
                              'input_std',
                              'input_mean_dx',
                              'input_std_dx',
                            #   'input_max_dx',
                              ],
                              axis='columns')
    df_train = df_train.dropna(axis='rows')
    df_train.to_csv('train.csv')

    return df_train

def plot_predictions(pred_ssim, pred_bitrate, true_ssim, true_bitrate):
    """
    Function to visualize predictions of pairs bitrate-ssim
    """
    
    metrics = list(pred_ssim.columns)
    asset = st.selectbox('Which asset to represent?', list(pred_ssim['source'].unique()))
  
    pred_ssim_df = pred_ssim[pred_ssim['source'] == asset]
    pred_bitrate_df = pred_bitrate[pred_bitrate['source'] == asset]

    resolutions = ['1080', '720', '480', '384', '288', '144']
    crfs = ['14', '18', '21', '25', '32', '40', '45']
    pred_plot_data = []

    x_pred = []
    y_pred = []
    x_true = []
    y_true = []

    for resolution in resolutions:
        for crf in crfs:
            ssim_feature = '{}_{}_ssim'.format(resolution, crf)
            bitrate_feature = '{}_{}_size'.format(resolution, crf)
            if '288_45' not in ssim_feature:
                x_pred.append(pred_bitrate_df[bitrate_feature].values[0])
                y_pred.append(pred_ssim_df[ssim_feature].values[0])
                x_true.append(true_bitrate[bitrate_feature].values[0])
                y_true.append(true_ssim[ssim_feature].values[0])

    pred_plot_data.append(go.Scatter(x=x_pred,
                                     y=y_pred,
                                     mode='markers',
                                     name='Predictions'))
    pred_plot_data.append(go.Scatter(x=x_true,
                                     y=y_true,
                                     mode='markers',
                                     name='True'))

    fig = go.Figure(data=pred_plot_data)

    st.plotly_chart(fig)

def main():
    """
    Main function
    """
    df_qoe = load_data(DATA_URI_QOE, 50000)
    df_brisque = load_data(DATA_URI_BRISQUE, 1000)

    st.write(df_qoe.head())
    st.write(df_brisque.head())

    plot_rd_curves(df_qoe)

    train_file_name = 'train.csv'
    if os.path.isfile(train_file_name):
        print('Train csv found')
        train_df = pd.read_csv(train_file_name)
    else:
        train_df = setup_train(df_qoe, df_brisque)
    st.write(train_df.head(100))

    pred_ssim, pred_bitrate, true_ssim, true_bitrate = predict_qoe(train_df)

    plot_predictions(pred_ssim, pred_bitrate, true_ssim, true_bitrate)

if __name__ == '__main__':

    main()
