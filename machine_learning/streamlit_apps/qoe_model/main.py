"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path

import pandas as pd
import streamlit as st
import numpy as np
import keras
from keras.models import Sequential
import plotly.express as px
import plotly.graph_objects as go

st.title('QoE model predictor')

DATA_URI = '../../cloud_functions/data-qoe-metrics-large.csv'

@st.cache
def load_data(nrows):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame.
    nrows limits the amount of data displayed for optimization
    """
    data_df = pd.read_csv(DATA_URI, nrows=nrows)
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

def set_rendition_name(x):
    """
    Function to extract source name from
    """
    try:
        return os.path.dirname(x).replace('/vimeo', '').split('/')[-1]
    except:
        return ''

def show_data():
    """
    Function to display the data in a text box in the app
    """
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data_df = load_data(1000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('Raw data')
    st.write(data_df)

    asset = st.selectbox('Which asset to represent?', list(data_df['title'].unique()))
    metric = st.selectbox('Which metric to display?', list(data_df.columns))
    rate_distortion_df = data_df[[metric, 'size', 'crf', 'rendition_ID', 'rendition']][data_df['title'] == asset]
    fig = px.scatter(rate_distortion_df, x='size', y=metric, color='rendition_ID', hover_data=['rendition'])
    st.plotly_chart(fig)

    
if __name__ == '__main__':
    show_data()

