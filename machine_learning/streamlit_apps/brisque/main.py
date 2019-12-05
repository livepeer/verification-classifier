"""
Module to generate an interactive app to visualize and train a QoE predictive model
using BRISQUE as input
It relies of Streamlite library for the visualization and display of widgets
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

st.title('QoE model predictor')

DATA_URI_BRISQUE = '../../cloud_functions/data-brisque-large.csv'
DATA_URI_QOE = '../../cloud_functions/data-qoe-metrics-large.csv'

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
    if 'qoe' in data_uri:
        data_df.rename(columns={'attack':'rendition', 'title':'source'}, inplace=True)
        data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
        data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.split('_')[0]))
        data_df['crf'] = data_df['rendition'].apply(lambda x: x.split('_')[-1])

    else:
        for column in data_df.columns:
                data_df[column] = data_df[column].astype(str).apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''),
                                                                                    dtype=np.float, sep=' '))
    return data_df

def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """

    return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]

def main():
    """
    Main function
    """
    df_qoe = load_data(DATA_URI_QOE, 5000)
    df_brisque = load_data(DATA_URI_BRISQUE, 1000)

    st.write(df_qoe)
    st.write(df_brisque)

if __name__ == '__main__':

    main()
