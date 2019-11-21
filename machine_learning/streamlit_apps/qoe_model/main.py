"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path

from catboost import Pool, CatBoostRegressor

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import fbeta_score, roc_curve, auc

st.title('QoE model predictor')

DATA_URI_QOE = '../../cloud_functions/data-qoe-metrics-large.csv'
DATA_URI_TAMPER = '../../cloud_functions/data-large.csv'
FEATURES = ["temporal_dct-max",
            "temporal_dct-euclidean",
            "temporal_dct-manhattan",
            "temporal_gaussian_mse-max",
            "temporal_gaussian_mse-manhattan",
            "temporal_gaussian_difference-mean",
            "temporal_gaussian_difference-max",
            "temporal_threshold_gaussian_difference-euclidean",
            "temporal_threshold_gaussian_difference-manhattan"
            ]
METRICS_QOE = 'temporal_ssim-mean'

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
    data_df.rename(columns={'attack':'rendition', 'title':'source'}, inplace=True)
    data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
    data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.split('_')[0]))
    if 'qoe' in data_uri:
        data_df['crf'] = data_df['rendition'].apply(lambda x: x.split('_')[-1])
        data_df['tamper'] = 1
    else:
        data_df['size_dimension_ratio'] = data_df['size'] / data_df['dimension_y']
        resolutions = ['1080', '720', '480', '360', '240', '144']
        data_df['tamper'] = data_df['rendition'].apply(lambda x: 1 if x in resolutions else -1)

    rendition_ids = list(data_df['rendition'].unique())
    data_df['rendition_ID'] = data_df['rendition'].apply(lambda x: set_rendition_id(x, rendition_ids))
    return data_df

def set_rendition_id(row, rendition_ids):
    """
    Function to assign ID numbers to renditions
    """
    return rendition_ids.index(row)

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


def unsupervised_evaluation(classifier, train_set, test_set, attack_set, beta=20):
    """
    Evaluates performance of unsupervised learning algorithms
    """
    y_pred_train = classifier.predict(train_set)
    y_pred_test = classifier.predict(test_set)
    y_pred_outliers = classifier.predict(attack_set)
    n_accurate_train = y_pred_train[y_pred_train == 1].size
    n_accurate_test = y_pred_test[y_pred_test == 1].size
    n_accurate_outliers = y_pred_outliers[y_pred_outliers == -1].size

    fpr, tpr, _ = roc_curve(np.concatenate([np.ones(y_pred_test.shape[0]),
                                            -1 * np.ones(y_pred_outliers.shape[0])]),
                            np.concatenate([y_pred_test, y_pred_outliers]),
                            pos_label=1)

    f_beta = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]),
                                         -1*np.ones(y_pred_outliers.shape[0])]),
                         np.concatenate([y_pred_test, y_pred_outliers]),
                         beta=beta,
                         pos_label=1)

    tnr = n_accurate_outliers / attack_set.shape[0]
    tpr_test = n_accurate_test / test_set.shape[0]
    tpr_train = n_accurate_train / train_set.shape[0]

    area = auc(fpr, tpr)
    return f_beta, area, tnr, tpr_train, tpr_test

def train_qoe_model(data_df):
    """
    Function to train model from given dataset
    """
    num_train = int(data_df.shape[0]*0.8)

    train_data = data_df.sample(num_train)
    test_data = data_df[~data_df.index.isin(train_data.index)]

    categorical_features_indices = []

    train_pool = Pool(data=train_data[FEATURES],
                      label=train_data[METRICS_QOE],
                      cat_features=categorical_features_indices)

    loss_funct = 'MAE'
    model_catbootregressor = CatBoostRegressor(depth=6,
                                               num_trees=150,
                                               l2_leaf_reg=5,
                                               learning_rate=0.05,
                                               loss_function=loss_funct
                                               )
    #train the model
    model_catbootregressor.fit(train_pool)

    return model_catbootregressor, test_data

def train_tamper_model(data_df):
    """
    Trains an unsupervised learning model for tamper verification
    """
    df_1 = data_df[data_df['tamper'] == 1]
    df_0 = data_df[~data_df.index.isin(df_1.index)]

    num_train = int(df_1.shape[0] * 0.8)
    df_train = df_1.sample(num_train)
    df_test = df_1[~df_1.index.isin(df_train.index)]

    df_train = df_1[0:num_train]
    df_test = df_1[num_train:]
    df_attacks = df_0.sample(frac=1)

    x_train = np.asarray(df_train[FEATURES])
    x_test = np.asarray(df_test[FEATURES])
    x_attacks = np.asarray(df_attacks[FEATURES])


    # Scaling the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_attacks = scaler.transform(x_attacks)

    oc_svm = svm.OneClassSVM(kernel='rbf', gamma=0.6, nu=0.002, cache_size=5000)

    oc_svm.fit(x_train)

    f_beta, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(oc_svm,
                                                                     x_train,
                                                                     x_test,
                                                                     x_attacks
                                                                     )
    st.write('F20:{} / Area:{} / TNR:{} / TPR:{}'.format(f_beta, area, tnr, tpr_test))

    return oc_svm, scaler, df_attacks

def main():
    """
    Main function to train and evaluate tamper and QoE models
    """
        # Get QoE pristine dataset (no attacks)
    df_qoe = load_data(DATA_URI_QOE, 50000)
    # Get tamper verification dataset (contains attacks)
    df_tamper = load_data(DATA_URI_TAMPER, 150000)
    # Remove low_bitrate kind of attacks
    df_tamper = df_tamper.loc[~df_tamper['rendition'].str.contains('low_bitrate')]

    # Merge datasets to train verification with more well encoded renditions
    frames = [df_qoe, df_tamper]
    df_aggregated = pd.concat(frames)

    # Display datasets
    st.subheader('Raw QoE data')
    st.write(df_qoe.head(100), df_qoe.shape)
    st.subheader('Raw tamper verification data')
    st.write(df_tamper.head(100), df_tamper.shape)
    st.subheader('Aggregated')
    st.write(df_aggregated.head(100), df_aggregated.shape)

    # Train SSIM predictor and retrieve a test set
    qoe_model, test_data_qoe = train_qoe_model(df_qoe)
    # Train tamper verification and extract
    tamper_model, scaler, df_attacks = train_tamper_model(df_aggregated)

    # Evaluate incidence of false positives from attack dataset
    x_attacks = np.asarray(df_attacks[FEATURES])
    x_attacks = scaler.transform(x_attacks)
    df_attacks['pred_tamper'] = tamper_model.predict(x_attacks)
    false_positives_df = df_attacks[df_attacks['tamper'] != df_attacks['pred_tamper']]
    st.write(false_positives_df.groupby('rendition').count())

    # Add predictions to test data set
    test_data_qoe['pred_ssim'] = qoe_model.predict(test_data_qoe[FEATURES])
    test_data_qoe['pred_tamper'] = tamper_model.predict(scaler.transform(test_data_qoe[FEATURES]))

    # Display correlation between predicted and measured metric and color them
    # according to their tamper classification
    fig = px.scatter(test_data_qoe,
                     x='pred_ssim',
                     y=METRICS_QOE,
                     color='pred_tamper',
                     hover_data=['rendition'])
    st.plotly_chart(fig)

    # Display True Positive Rate for the test dataset
    st.write(test_data_qoe[test_data_qoe['pred_tamper'] > 0].shape[0] / test_data_qoe.shape[0])

if __name__ == '__main__':

    main()
