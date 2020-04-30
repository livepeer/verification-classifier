"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path
import json

from catboost import Pool, CatBoostRegressor
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title('QOE model training environment')

DATA_URI_QOE = '../../cloud_functions/data-qoe-metrics-large.csv'

FEATURES = ['size_dimension_ratio',
            'dimension',
            'size',
            'temporal_dct-mean',
            'temporal_gaussian_mse-mean',
            'temporal_gaussian_difference-mean',
            'temporal_threshold_gaussian_difference-mean'
            ]

METRICS_QOE = ['temporal_ssim-mean']

def load_data(data_uri, nrows):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame suitable for model training.
    nrows limits the amount of data displayed for optimization
    """
    data_df = pd.read_csv(data_uri, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data_df.rename(lowercase, axis='columns', inplace=True)
    data_df.rename(columns={'attack':'rendition', 'title':'source'}, inplace=True)
    data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
    data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.split('_')[0]))
    data_df['crf'] = data_df['rendition'].apply(lambda x: x.split('_')[-1])

    return data_df


def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """
    try:
        if 'p' in rendition_name:
            rendition_name = rendition_name.replace('p', '')
        return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]
    except:
        return ''


def train_qoe_model(data_df):
    """
    Function to train model from given dataset
    """
    st.write('QOE MODEL SOURCE DATASET', data_df.head(100))

    num_train = int(data_df.shape[0] * 0.8)

    train_data = data_df.sample(num_train)
    test_data = data_df[~data_df.index.isin(train_data.index)]

    categorical_features_indices = []

    train_pool = Pool(data=train_data[FEATURES],
                      label=train_data[METRICS_QOE],
                      cat_features=categorical_features_indices)

    loss_funct = 'MAE'
    CB_params = {'depth':6,
                 'num_trees':100,
                 'l2_leaf_reg':5,
                 'learning_rate':0.05,
                 'loss_function':loss_funct}
    model_catbootregressor = CatBoostRegressor(**CB_params)

    #Train the model
    print('Training QoE model:')
    model_catbootregressor.fit(train_pool)

    test_pool = Pool(data=test_data[FEATURES],
                     label=test_data[METRICS_QOE],
                     cat_features=categorical_features_indices)
    learn_test_df = pd.DataFrame(model_catbootregressor.eval_metrics(test_pool,
                                                                     ['MAE', 'MAPE', 'RMSE']))

    st.write('QoE model test set accuracy:')
    st.write(learn_test_df.min())

    model_catbootregressor.save_model('CB_Regressor.cbm',
                                      format="cbm",
                                      export_parameters=None,
                                      pool=None)

    CB_params['eval_metrics'] = learn_test_df.min().to_json()
    CB_params['features'] = FEATURES
    with open('param_CB_Regressor.json', 'w') as outfile:
        json.dump(CB_params, outfile)

    return model_catbootregressor

def plot_scatter(title, x_metric, y_metric, x_axis_title, y_axis_title, df_aggregated, line=False):
    """
    Function to plot and format a scatterplot from the aggregated dataframe
    """
    resolutions = list(df_aggregated['dimension_y'].unique())
    resolutions.sort()
    data = []
    shapes = list()

    for res in resolutions:
        data.append(go.Scatter(x=df_aggregated[df_aggregated['dimension_y'] == res][x_metric],
                               y=df_aggregated[df_aggregated['dimension_y'] == res][y_metric],
                               mode='markers',
                               marker=dict(color=res,
                                           opacity=0.8,
                                           line=dict(width=0)
                                           ),
                               name=str(res)
                               )
                    )

    if line:
        trace_line = go.Scatter(x=np.arange(0, 1.1, 0.1),
                                y=np.arange(0, 1.1, 0.1),
                                mode='lines',
                                name='y=x')
        data.append(trace_line)
    else:
        shapes.append({'type': 'line',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': 0,
                       'y0': 0,
                       'x1': 0,
                       'y1': 1000})

    fig = go.Figure(data=data)
    fig.update_layout(title=title,
                      xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title,
                      legend=go.layout.Legend(x=0,
                                              y=1,
                                              traceorder="normal",
                                              font=dict(family="sans-serif",
                                                        size=12,
                                                        color="black"
                                                       ),
                                              bgcolor="LightSteelBlue",
                                              bordercolor="Black",
                                              borderwidth=2
                                              ),
                      shapes=shapes
                     )
    st.plotly_chart(fig, width=1000, height=1000)

def plot_histogram(metric, x_title, df_aggregated):
    """
    Function to plot and format a histogram from the aggregated dataframe
    """
    resolutions = list(df_aggregated['dimension_y'].unique())
    resolutions.sort()
    data = []
    for res in resolutions:
        data.append(go.Histogram(x=df_aggregated[metric][df_aggregated['dimension_y'] == res],
                                 name='{}p'.format(res),
                                 autobinx=False,
                                 nbinsx=500,
                                 opacity=0.75))
    shapes = list()
    shapes.append({'type': 'line',
                   'xref': 'x',
                   'yref': 'y',
                   'x0': 0,
                   'y0': 0,
                   'x1': 0,
                   'y1': 1000})

    fig = go.Figure(data=data)
    fig.layout.update(barmode='overlay',
                      title='Histogram of legit assets',
                      xaxis_title_text=x_title,
                      yaxis_title_text='Count',
                      legend=go.layout.Legend(x=1,
                                              y=1,
                                              traceorder="normal",
                                              font=dict(family="sans-serif",
                                                       size=12,
                                                       color="black"
                                                      )
                                             ),
                      shapes=shapes
                      )
    st.plotly_chart(fig)

def plot_correlation_matrix(df_aggregated):
    """
    Display correlation matrix for features
    """
    df_features = pd.DataFrame(df_aggregated[FEATURES + METRICS_QOE])
    corr = df_features.corr()
    corr.style.background_gradient(cmap='coolwarm')
    fig = go.Figure(data=go.Heatmap(x=FEATURES + METRICS_QOE,
                                    y=FEATURES + METRICS_QOE,
                                    z=corr
                                    ))
    st.plotly_chart(fig, width=1000, height=1000)

def main():
    """
    Main function to train and evaluate QoE models
    """
    # Get QoE pristine dataset (no attacks)
    df_qoe = load_data(DATA_URI_QOE, 50000)

    # Display datasets
    st.subheader('Raw QoE data')
    st.write(df_qoe[FEATURES + ['path']].head(100), df_qoe.shape)

    qoe_columns = FEATURES + METRICS_QOE

    #######################################################################################
    # Train SSIM predictor and add predictions to its dataframe
    #######################################################################################
    qoe_model = train_qoe_model(df_qoe[qoe_columns])
    df_qoe['pred_ssim'] = qoe_model.predict(df_qoe[FEATURES])

    # Output the error check to the frontend
    df_ssim = df_qoe[['pred_ssim', 'temporal_ssim-mean']].dropna(axis='rows')
    rmse = (df_ssim['pred_ssim'] - df_ssim['temporal_ssim-mean']) ** 2
    mape = np.abs(df_ssim['temporal_ssim-mean'] - df_ssim['pred_ssim']) / df_ssim['temporal_ssim-mean']
    st.write('RMSE:', rmse.mean() ** .5)
    st.write('MAPE:', mape.mean())

    #######################################################################################
    # Display plots
    #######################################################################################

    df_plots_qoe = df_qoe.sample(n=5000, random_state=1)

    # Display difference between predicted ssim and measured ssim
    plot_scatter('Measured SSIM vs Predicted SSIM',
                 'pred_ssim',
                 'temporal_ssim-mean',
                 'Predicted SSIM',
                 'SSIM',
                 df_plots_qoe,
                 line=True)
    # Display correlation matrix
    plot_correlation_matrix(df_plots_qoe)

if __name__ == '__main__':

    main()
