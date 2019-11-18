"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path

from catboost import Pool, CatBoostRegressor
import catboost as cgb
import xgboost as xgb

import pandas as pd
import pandas_profiling
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

st.title('QoE model predictor')

DATA_URI = '../../cloud_functions/data-qoe-metrics-large.csv'

def create_report():
    data_df = pd.read_csv(DATA_URI)
    profile = data_df.profile_report(title='Pandas Profiling Report')
    profile.to_file(output_file="output.html")

@st.cache
def load_data(nrows):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame.
    nrows limits the amount of data displayed for optimization
    """
    data_df = pd.read_csv(DATA_URI, nrows=nrows)
    print(data_df.shape)
    lowercase = lambda x: str(x).lower()
    data_df.rename(lowercase, axis='columns', inplace=True)
    data_df.rename(columns={'attack':'rendition'}, inplace=True)
    data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
    data_df['crf'] = data_df['rendition'].apply(lambda x: x.split('_')[-1])
    data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.split('_')[0]))
    data_df['size'] = data_df['size_dimension_ratio'] * data_df['dimension_y']
    rendition_ids = list(data_df['rendition'].unique())
    print(rendition_ids)
    data_df['rendition_ID'] = data_df['rendition'].apply(lambda x: set_rendition_id(x, rendition_ids))
    return data_df

def set_rendition_id(x, rendition_ids):
    """
    Function to assign ID numbers to renditions
    """
    return rendition_ids.index(x)

def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """
    try:
        return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]
    except:
        return ''

def show_data():
    """
    Function to display the data in a text box in the app
    """
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data_df = load_data(100)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('Raw data')
    st.write(data_df)

    asset = st.selectbox('Which asset to represent?', list(data_df['title'].unique()))
    metric = st.selectbox('Which metric to display?', list(data_df.columns))
    rate_distortion_features = [metric, 'size', 'crf', 'rendition_ID', 'rendition']
    rate_distortion_df = data_df[rate_distortion_features][data_df['title'] == asset]
    st.write(rate_distortion_df[['size', metric]])
    points_2d = rate_distortion_df[['size', metric]].values
    points_df = rate_distortion_df[['size', metric]].iloc[ConvexHull(points_2d).vertices]
    points_df = points_df.sort_values(by='size')
    x_values = points_df['size'].values
    y_values = points_df[metric].values

    cubic_spline = CubicSpline(x_values, y_values)
    x_interpolation = np.arange(0, max(points_df['size']), 100)

    interpolation_df = pd.DataFrame()
    interpolation_df['x'] = x_interpolation
    interpolation_df['y'] = cubic_spline(x_interpolation)
    print(cubic_spline.c.shape)
    fig = go.Figure(go.Scatter(x=interpolation_df['x'],
                               y=interpolation_df['y'],
                               name='Interpolation',
                               showlegend=True))

    fig.add_trace(go.Scatter(x=points_df['size'],
                             y=points_df[metric],
                             name='Original',
                             showlegend=True))

    fig.update_layout(title="Convex hull with Cubic Spline interpolation")
    st.plotly_chart(fig)
    fig = px.scatter(rate_distortion_df,
                     x='size',
                     y=metric,
                     color='rendition_ID',
                     hover_data=['rendition'])
    st.plotly_chart(fig)

def get_convex_hull(row, metric):
    """
    Retrieves a list of point coordinates representing the 
    convex hull of a set of points from a pd.DataFrame row
    """
    points_2d = row[['size', metric]].values
    print(points_2d)
    points_df = row[['size', metric]].iloc[ConvexHull(points_2d).vertices]
    points_df = points_df.sort_values(by='size')
    return points_df

def cat_hyp(depth, bagging_temperature):
    """
    Function to optimize depth and bagging temperature of a catboost regressor
    """

    params = {"iterations": 100,
             "learning_rate": 0.025,
             "eval_metric": "R2",
             "verbose": False} # Default Parameters
    params["depth"] = int(round(depth))
    params["bagging_temperature"] = bagging_temperature

    # specify the training parameters 

    scores = cgb.cv(train_pool,
                    params,
                    fold_count=3)
    return np.max(scores['test-R2-mean'])  # Return maximum R-Squared value     
  
def prepare_data(n_rows):
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    data_df = load_data(n_rows)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('Raw data')
    st.write(data_df.head())

    metric = 'temporal_ssim-mean'
    data_df['convex_hull'] = data_df.apply(lambda row: get_convex_hull(row, metric), axis=1)
    st.write(data_df.head())
    # return data_df

def train_models():
    """
    Function to train model from given dataset
    """
    # initialize data

    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('Raw data')
    st.write(data_df.head())

    # Search space
    pds = {'depth': (5,8),
           'bagging_temperature': (3, 10)
           }
    # Surrogate model
    optimizer = BayesianOptimization(cat_hyp, pds, random_state=2100)
                                    
    # Optimize
    optimizer.maximize(init_points=3, n_iter=7)
    print(optimizer.max['params']['depth'])
    model_catbootregressor = CatBoostRegressor(iterations=1000,
                                               depth=int(optimizer.max['params']['depth']),
                                               bagging_temperature=float(optimizer.max['params']['bagging_temperature']),
                                               learning_rate=0.005,
                                               loss_function='RMSE')
    #train the model
    model_catbootregressor.fit(train_pool)
    # make the prediction using the resulting model
    test_data['preds_CBR'] = model_catbootregressor.predict(test_pool)

    st.write(test_data[['preds_CBR', metric]].describe())

    rmse_catbootregressor = np.sqrt(mean_squared_error(test_data[metric], test_data['preds_CBR']))

    print("RMSE CatBoost: %f" % (rmse_catbootregressor))

if __name__ == '__main__':
    show_data()
    # prepare_data()
    # if st.button('Report'):
    #     create_report()
    # else:

    metric = 'temporal_ssim-mean'
    features = [#"dimension_y",
                "size",
                "temporal_dct-mean",
                "temporal_gaussian_mse-mean",
                "temporal_gaussian_difference-mean",
                "temporal_threshold_gaussian_difference-mean",
                # "rendition_ID"
                ]
    

    data_df = load_data(50000)
    num_train = int(data_df.shape[0]*0.8)

    train_data = data_df[0:num_train]
    train_label = data_df[0:num_train]
    test_data = data_df[num_train:]

    categorical_features_indices = []#[0, 6]

    train_pool = Pool(data=train_data[features],
                      label=train_label[metric],
                      cat_features=categorical_features_indices)
    test_pool = Pool(test_data[features],
                     cat_features=categorical_features_indices)
    train_models()
