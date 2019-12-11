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
    # row['max_dx'] = np.max(np.diff(features, axis=0), axis=0)

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
    metric_y = st.selectbox('Which metric to represent for Y?', metrics)

    rate_distortion_df = df_qoe[[metric_x, metric_y, 'dimension_y', 'rendition']][df_qoe['source'] == asset]

    data = []
    dimensions = rate_distortion_df['dimension_y'].unique()
    dimensions.sort()
    for dimension in dimensions:

        trace = go.Scatter(x=rate_distortion_df[rate_distortion_df['dimension_y']==dimension][metric_x],
                           y=rate_distortion_df[rate_distortion_df['dimension_y']==dimension][metric_y],
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

def train_qoe_model(data_df):
    """
    Function to train model from given dataset
    """
    num_train = int(data_df.shape[0] * 0.8)

    train_data = data_df.sample(num_train)
    test_data = data_df[~data_df.index.isin(train_data.index)]

    x_features = [feature for feature in data_df.columns if 'input' in feature]
    x_features.append('dimension_y')
    # x_features.append('288_14_ssim')
    # x_features.append('288_14_size')
    # x_features.append('288_14_crf')
    x_features.append('1080_14_size')
    x_features.append('1080_14_ssim')
    x_features.append('1080_14_pixels')
    y_features = [feature for feature in data_df.columns if 'size' in feature]
    # y_features.remove('288_14_size')
    y_features.remove('1080_14_size')
    y_features.sort()

    st.write(train_data[x_features].head())
    st.write(train_data[y_features].head())

    x_train = np.asarray(train_data[x_features])
    x_test = np.asarray(test_data[x_features])
    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_features = ['1080_18_size']

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(train_data[y_features].values)
    y_test = y_scaler.transform(test_data[y_features].values)

    learn_mape_test_df = pd.DataFrame()
    learn_rmse_test_df = pd.DataFrame()
    learn_mape_train_df = pd.DataFrame()
    learn_rmse_train_df = pd.DataFrame()

    models_dict = dict()
    pools_dict = dict()

    categorical_features_indices = [x_features.index('crf')]
    x_train = pd.DataFrame(x_train, columns=x_features)
    x_test = pd.DataFrame(x_test, columns=x_features)

    x_train['crf'] = train_data['crf'].astype('str')
    x_test['crf'] = test_data['crf'].astype('str')

    print(x_train['crf'].unique())
    for feature in y_features:
        print(feature)

        y_train = pd.DataFrame(data=y_train, columns=y_features)
        y_test = pd.DataFrame(data=y_test, columns=y_features)

        train_pool = Pool(data=x_train,
                          label=y_train[feature],
                          cat_features=categorical_features_indices)

        if 'size' in feature:
            loss_funct = 'RMSE'
            num_trees = 500
        else:
            loss_funct = 'RMSE'
            num_trees = 100
        models_dict[feature] = CatBoostRegressor(depth=3,
                                                 num_trees=num_trees,
                                                 l2_leaf_reg=2,
                                                 learning_rate=0.05,
                                                 loss_function=loss_funct,
                                                 bagging_temperature=10
                                                 )
        #train the model
        print('Training QoE model:')
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
    st.write('Feature:', y_features[0])
    fig = go.Figure(data=[go.Scatter(x=learn_mape_test_df.index, y=learn_mape_test_df[y_features[0]], name='Test'),
                          go.Scatter(x=learn_mape_test_df.index, y=learn_mape_train_df[y_features[0]], name='Train')])
    st.plotly_chart(fig)
    st.write('QoE model test set RMSE:')
    st.write(learn_rmse_test_df.min())
    fig = go.Figure(data=[go.Scatter(x=learn_rmse_test_df.index, y=learn_rmse_test_df[y_features[0]], name='Test'),
                          go.Scatter(x=learn_rmse_test_df.index, y=learn_rmse_train_df[y_features[0]], name='Train')])
    st.plotly_chart(fig)

    y_pred = models_dict[y_features[0]].predict(pools_dict[y_features[0]])
    
    fig = go.Figure(data=[go.Histogram(x=y_test[y_features[0]]),
                          go.Histogram(x=y_pred)])
    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    st.plotly_chart(fig)
    return models_dict

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
                              'source'],
                              axis='columns')
    df_train = df_train.dropna(axis='rows')
    df_train.to_csv('train.csv')

    return df_train

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

    train_qoe_model(train_df)

if __name__ == '__main__':

    main()
