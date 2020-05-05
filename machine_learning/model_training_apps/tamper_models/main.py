"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path
import json
import sys

from catboost import CatBoostClassifier
from joblib import dump
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import fbeta_score

sys.path.insert(0, '../../../scripts/asset_processor')
from video_asset_processor import VideoAssetProcessor

DATA_URI_TAMPER = '../../cloud_functions/data-large.csv'

FEATURES = ['size_dimension_ratio',
            'temporal_dct-mean',
            'temporal_gaussian_mse-mean',
            'temporal_gaussian_difference-mean',
            'temporal_threshold_gaussian_difference-mean'
            ]

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

    data_df['size_dimension_ratio'] = data_df['size'] / data_df['dimension']
    resolutions = ['1080', '720', '480', '360', '240', '144']
    data_df['tamper'] = data_df['rendition'].apply(lambda x: 1 if x in resolutions else -1)

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

def ul_model_evaluation(classifier, test_set, attack_set, beta=20):
    """
    Evaluates performance of supervised and unsupervised learning algorithms
    """
    y_pred_test = classifier.predict(test_set).astype(float)
    y_pred_outliers = classifier.predict(attack_set).astype(float)

    n_accurate_test = y_pred_test[y_pred_test == 1].size
    n_accurate_outliers = y_pred_outliers[y_pred_outliers != 1].size

    f_beta = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]),
                                         -1 * np.ones(y_pred_outliers.shape[0])]),
                         np.concatenate([y_pred_test, y_pred_outliers]),
                         beta=beta,
                         pos_label=1)

    tnr = n_accurate_outliers / attack_set.shape[0]
    tpr_test = n_accurate_test / test_set.shape[0]

    return f_beta, tnr, tpr_test

def sl_model_evaluation(classifier, eval_df):
    """
    Evaluates performance of supervised and unsupervised learning algorithms
    """

    untampered_df = eval_df[eval_df['tamper'] == 1]
    attacks_df = eval_df[eval_df['tamper'] == -1]

    y_pred_test = classifier.predict(np.asarray(untampered_df[FEATURES]))
    y_pred_outliers = classifier.predict(np.asarray(attacks_df[FEATURES]))

    # Format the output of the classifier
    y_pred_test[y_pred_test == 0] = -1
    y_pred_outliers[y_pred_outliers == 0] = -1

    n_accurate_test = y_pred_test[y_pred_test == 1].size
    n_accurate_outliers = y_pred_outliers[y_pred_outliers == -1].size

    f_beta = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]),
                                         -1 * np.ones(y_pred_outliers.shape[0])]),
                         np.concatenate([y_pred_test, y_pred_outliers]),
                         beta=20,
                         pos_label=1)

    tnr = n_accurate_outliers / attacks_df.shape[0]
    tpr_test = n_accurate_test / untampered_df.shape[0]

    return f_beta, tnr, tpr_test

def meta_model_evaluation(data_df):
    """
    Evaluate performance of combined meta-model
    """
    eval_df = data_df

    attacks_df = eval_df[eval_df['tamper'] == -1]
    untampered_df = eval_df[eval_df['tamper'] == 1]

    y_pred_test = untampered_df['meta_pred_tamper']
    y_pred_outliers = attacks_df['meta_pred_tamper']

    f_beta = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]),
                                         -1*np.ones(y_pred_outliers.shape[0])]),
                         np.concatenate([y_pred_test, y_pred_outliers]),
                         beta=20,
                         pos_label=1)

    try:
        tnr = attacks_df[attacks_df['meta_pred_tamper'] == -1].shape[0] / attacks_df.shape[0]
    except:
        tnr = 0
        print('No attacks')

    tpr = untampered_df[untampered_df['meta_pred_tamper'] == 1].shape[0] / untampered_df.shape[0]

    st.write('TAMPER META-MODEL ACCURACY')
    st.write('F20: {} / TNR: {} / TPR: {}'.format(f_beta, tnr, tpr))

    st.write('MODEL ANALYSIS')
    st.subheader('Unsupervised model false positives')
    st.write('Shape of attacks:', attacks_df.shape)
    st.write(attacks_df[attacks_df['ul_pred_tamper'] == 1].groupby('rendition').count())
    st.subheader('Supervised model false positives')
    st.write(attacks_df[attacks_df['sl_pred_tamper'] == 1].groupby('rendition').count())
    st.subheader('Meta model false positives')
    st.write(attacks_df[attacks_df['meta_pred_tamper'] == 1].groupby('rendition').count())

    st.subheader('Unsupervised model false negatives')
    st.write('Shape of untampered', untampered_df.shape)
    st.write(untampered_df[untampered_df['ul_pred_tamper'] == -1].groupby('rendition').count())
    st.subheader('Supervised model false negatives')
    st.write(untampered_df[untampered_df['sl_pred_tamper'] == -1].groupby('rendition').count())
    st.subheader('Meta model false negatives')
    st.write(untampered_df[untampered_df['meta_pred_tamper'] == -1].groupby('rendition').count())

def meta_model(row):
    """
    Inputs the metamodel AND operator as condition
    Retrieves the tamper value of the UL model only when both models agree in classifying
    as non tampered. Otherwise retrieves the SL classification
    UL classifier has a higher TPR but lower TNR, meaning it is less restrictive towards
    tampered assets. SL classifier has higher TNR but is too punitive, which is undesirable,
    plus it requires labeled data.
    """
    meta_condition = row['ul_pred_tamper'] == 1 and row['sl_pred_tamper'] == 1
    if meta_condition:
        return row['ul_pred_tamper']
    return row['sl_pred_tamper']

def train_ul_tamper_model(data_df):
    """
    Trains an unsupervised learning model for tamper verification
    """
    # Output the used dataset to confirm that uses the same as Supervised learning
    st.write('UL TAMPER SOURCE DATASET', data_df.head(100))
    st.write('Total samples:', data_df.shape)

    # Get tampered (attacks) dataset
    df_attacks = data_df[data_df['tamper'] == -1].sample(frac=1)
    df_untampered = data_df[data_df['tamper'] == 1].sample(frac=1)
    # Split dataset into train, test and attacks and shuffle them
    df_train_ul = df_untampered.sample(frac=0.8)
    df_test_ul = df_untampered[~df_untampered.index.isin(df_train_ul.index)]
    st.write('Train / test split:', df_train_ul.shape, df_test_ul.shape)
    # Convert datasets from dataframes to numpy arrays
    x_train_ul = np.asarray(df_train_ul[FEATURES])
    x_test_ul = np.asarray(df_test_ul[FEATURES])
    x_attacks = np.asarray(df_attacks[FEATURES])

    # Scale the data
    scaler = StandardScaler()
    x_train_ul = scaler.fit_transform(x_train_ul)
    x_test_ul = scaler.transform(x_test_ul)
    x_attacks = scaler.transform(x_attacks)

    # Define One Class Support Vector Machine model and train it
    oc_svm = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=0.01, cache_size=5000)
    oc_svm.fit(x_train_ul)

    # Evaluate its accuracy
    f_beta, tnr, tpr_test = ul_model_evaluation(oc_svm,
                                                x_test_ul,
                                                x_attacks
                                                )

    # Save the scaler for inference
    dump(scaler, '../../output/models/UL_StandardScaler.joblib')
    # Save the OC-SVM for inference
    dump(oc_svm, '../../output/models/OCSVM.joblib')
    svm_params = oc_svm.get_params()
    svm_params['features'] = FEATURES
    svm_params['f_beta'] = f_beta
    svm_params['tnr'] = tnr
    svm_params['tpr_test'] = tpr_test
    with open('../../output/models/param_OCSVM.json', 'w') as outputfile:
        json.dump(svm_params, outputfile)


    st.write('UNSUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / TNR:{} / TPR_test:{}'.format(f_beta,
                                                    tnr,
                                                    tpr_test))

    return oc_svm, scaler

def train_sl_tamper_model(data_df):
    """
    Trains a supervised learning model for tamper verification using a Catboost classifier
    """
    # Get untampered assets dataset
    # Output the used dataset to confirm that uses the same as Supervised learning
    st.write('SL TAMPER SOURCE DATASET', data_df.head(100))
    st.write('Total samples:', data_df.shape)

    # Get tampered (attacks) dataset
    # Split dataset into train, test and attacks and shuffle them
    df_train_sl = data_df.sample(frac=0.8)
    df_test_sl = data_df[~data_df.index.isin(df_train_sl.index)]

    # Remove types of attacks from training that are too obvious to reduce bias
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('black_and_white')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('rotate')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vignette')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vertical')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('345x114')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('856x856')]

    y_train_sl = df_train_sl['tamper']

    cat_features = []
    # Initialize CatBoostClassifier
    cb_params = dict(iterations=500,
                     learning_rate=0.05,
                     depth=6)
    catboost_binary = CatBoostClassifier(**cb_params)
    # Fit model
    catboost_binary.fit(np.asarray(df_train_sl[FEATURES]), y_train_sl, cat_features)

    # Evaluate its accuracy
    f_beta, tnr, tpr_test = sl_model_evaluation(catboost_binary, df_test_sl)
    st.write('SUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / TNR:{} / TPR_test:{}'.format(f_beta,
                                                    tnr,
                                                    tpr_test))

    catboost_binary.save_model('../../output/models/CB_Binary.cbm',
                               format="cbm",
                               export_parameters=None,
                               pool=None)

    cb_params['eval_metrics'] = {'f_beta':f_beta, 'tnr':tnr, 'tpr_test':tpr_test}
    cb_params['features'] = FEATURES
    with open('../../output/models/param_CB_Binary.json', 'w') as outfile:
        json.dump(cb_params, outfile)

    return catboost_binary

def plot_3d(title, z_metric, z_axis, color_metric, df_aggregated):
    """
    Function to plot and format a 3D scatterplot from the aggregated dataframe
    """

    fig = go.Figure(data=go.Scatter3d(x=df_aggregated['dimension_y'],
                                      y=df_aggregated['size'],
                                      z=df_aggregated[z_metric],
                                      mode='markers',
                                      marker=dict(size=1,
                                                  color=df_aggregated[color_metric],
                                                  opacity=0.8
                                                 )
                                      ))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title="Vertical Resolution",
                                 yaxis_title="File Size",
                                 zaxis_title=z_axis),
                      font=dict(size=15),
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
                                             )
                     )
    st.plotly_chart(fig, width=1000, height=1000)

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


def main():
    """
    Main function to train and evaluate tamper models
    """

    # Get tamper verification dataset (contains attacks)
    df_tamper = load_data(DATA_URI_TAMPER, None)

    df_tamper = VideoAssetProcessor.rescale_to_resolution(df_tamper, FEATURES_UL)
    # Display dataset
    st.subheader('Raw tamper verification data')
    st.write(df_tamper[FEATURES + ['path', 'tamper']].head(100), df_tamper.shape)

    st.write('Dataframe available features:', df_tamper.columns)
    st.write('Dataframe considered attacks', df_tamper['rendition'].unique())

    ul_columns = FEATURES + ['tamper', 'rendition', 'path']
    sl_columns = FEATURES + ['tamper', 'rendition', 'path']


    #######################################################################################
    # Train unsupervised tamper verification model and retrieve its feature scaler
    #######################################################################################
    oc_svm, scaler = train_ul_tamper_model(df_tamper[ul_columns])

    # Add predictions to tamper dataframe
    x_test = scaler.transform(df_tamper[FEATURES])
    df_tamper['ocsvm_dist'] = oc_svm.decision_function(x_test)
    df_tamper['ul_pred_tamper'] = oc_svm.predict(x_test)


    #######################################################################################
    # Train supervised tamper verification and adds its predictions to dataframe
    #######################################################################################
    catboost_binary = train_sl_tamper_model(df_tamper[sl_columns])
    df_tamper['sl_pred_tamper'] = catboost_binary.predict(df_tamper[FEATURES])
    df_tamper['sl_pred_tamper'] = df_tamper['sl_pred_tamper'].apply(lambda x: 1 if x == 1 else -1)

    #######################################################################################
    # Apply meta-model to aggregated dataset
    #######################################################################################
    df_tamper['meta_pred_tamper'] = df_tamper.apply(meta_model, axis=1)

    st.write('FINAL DF:', df_tamper[FEATURES + ['rendition',
                                                'sl_pred_tamper',
                                                'ul_pred_tamper',
                                                'meta_pred_tamper',
                                                'path']].head(100))

    # Display evaluation metrics for the test dataset
    meta_model_evaluation(df_tamper)

    #######################################################################################
    # Display plots
    #######################################################################################
    df_plots_aggregated = df_tamper.sample(n=5000, random_state=1)
    # Display correlation between measured distance to decision function, resolution and size.
    plot_3d('OC-SVM Classifier',
            'ocsvm_dist',
            'Distance to Decision Function',
            'tamper',
            df_plots_aggregated)
    # Display histogram of non-tampered assets
    plot_histogram('ocsvm_dist', 'Distance to decision function', df_tamper)

if __name__ == '__main__':

    main()
