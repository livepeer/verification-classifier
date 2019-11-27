"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path

from catboost import Pool, CatBoostRegressor, CatBoostClassifier

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import fbeta_score, roc_curve, auc

st.title('QoE model predictor')

DATA_URI_QOE = '../../cloud_functions/data-qoe-metrics-large.csv'
DATA_URI_TAMPER = '../../cloud_functions/data-large.csv'
FEATURES = ['temporal_dct-max',
            'temporal_dct-euclidean',
            'temporal_dct-manhattan',
            'temporal_gaussian_mse-max',
            'temporal_gaussian_mse-manhattan',
            'temporal_gaussian_difference-mean',
            'temporal_gaussian_difference-max',
            'temporal_threshold_gaussian_difference-euclidean',
            'temporal_threshold_gaussian_difference-manhattan',
            # 'pred_ssim',
            'size_dimension_ratio'
            ]
FEATURES_SL = ['temporal_dct-max',
               'temporal_dct-euclidean',
               'temporal_dct-manhattan',
               'temporal_gaussian_mse-max',
               'temporal_gaussian_mse-manhattan',
               'temporal_gaussian_difference-mean',
               'temporal_gaussian_difference-max',
               'temporal_threshold_gaussian_difference-euclidean',
               'temporal_threshold_gaussian_difference-manhattan',
               'size_dimension_ratio',
            #    'pred_ssim'
               ]
FEATURES_QOE = ['temporal_dct-max',
               'temporal_dct-mean',
               'temporal_dct-std',
               'temporal_dct-euclidean',
               'temporal_dct-manhattan',
               'temporal_gaussian_mse-max',
               'temporal_gaussian_mse-mean',
               'temporal_gaussian_mse-std',
               'temporal_gaussian_mse-euclidean',
               'temporal_gaussian_mse-manhattan',
               'temporal_gaussian_difference-max',
               'temporal_gaussian_difference-mean',
               'temporal_gaussian_difference-std',
               'temporal_gaussian_difference-euclidean',
               'temporal_gaussian_difference-manhattan',
               'temporal_threshold_gaussian_difference-max',
               'temporal_threshold_gaussian_difference-mean',
               'temporal_threshold_gaussian_difference-std',
               'temporal_threshold_gaussian_difference-euclidean',
               'temporal_threshold_gaussian_difference-manhattan',
               'size_dimension_ratio'
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
        data_df['size_dimension_ratio'] = data_df['size'] / data_df['dimension_y'] * data_df['dimension'] * data_df['fps']
        resolutions = ['1080', '720', '480', '360', '240', '144']
        data_df['tamper'] = data_df['rendition'].apply(lambda x: 1 if x in resolutions else -1)

    return data_df

def rescale_to_resolution(data):
    """
    Function to rescale features to improve accuracy
    """
    df = pd.DataFrame(data)
    print(list(df.columns))
    features = list(df.columns)
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
        downscale_feature = [feature for feature in features if label in feature]
        if downscale_feature:
            for feature in downscale_feature:
                if df[feature].dtype == int or df[feature].dtype == float:
                    print('Downscaling', label, feature)
                    df[feature] = df[feature] / df['size_dimension_ratio']

    for label in upscale_features:
        upscale_feature = [feature for feature in features if label in feature]
        if upscale_feature:
            for feature in upscale_feature:
                if df[feature].dtype == int or df[feature].dtype == float:
                    print('Upscaling', label, feature)
                    df[feature] = df[feature] * df['size_dimension_ratio']

    return df

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


def model_evaluation(classifier, train_set, test_set, attack_set, beta=20):
    """
    Evaluates performance of supervised and unsupervised learning algorithms
    """
    y_pred_train = classifier.predict(train_set).astype(float)
    y_pred_test = classifier.predict(test_set).astype(float)
    y_pred_outliers = classifier.predict(attack_set).astype(float)

    y_pred_train[y_pred_train == 0] = -1
    y_pred_test[y_pred_test == 0] = -1
    y_pred_outliers[y_pred_outliers == 0] = -1

    n_accurate_train = y_pred_train[y_pred_train == 1].size
    n_accurate_test = y_pred_test[y_pred_test == 1].size
    n_accurate_outliers = y_pred_outliers[y_pred_outliers != 1].size

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

def meta_model(row):
    return row['sl_pred_tamper'] if row['sl_pred_tamper'] == -1 else row['ul_pred_tamper']

def meta_model_evaluation(data_df, sl_classifier, ul_classifier, scaler):
    """
    Evaluate performance of combined meta-model
    """
    eval_df = data_df

    scaled_input = scaler.transform(np.asarray(eval_df[FEATURES]))
    eval_df['ul_pred_tamper'] = ul_classifier.predict(scaled_input)
    eval_df['sl_pred_tamper'] = sl_classifier.predict(eval_df[FEATURES])
    eval_df['sl_pred_tamper'] = eval_df['sl_pred_tamper'].apply(lambda x: 1 if  x == 1 else -1)

    eval_df['meta_pred_tamper'] = eval_df.apply(meta_model, axis=1)
    attacks_df = eval_df[eval_df['tamper'] == -1]
    untampered_df = eval_df[eval_df['tamper'] == 1]

    y_pred_test = untampered_df['meta_pred_tamper']
    y_pred_outliers = attacks_df['meta_pred_tamper']

    f_beta = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]),
                                         -1*np.ones(y_pred_outliers.shape[0])]),
                         np.concatenate([y_pred_test, y_pred_outliers]),
                         beta=20,
                         pos_label=1)

    tnr = attacks_df[attacks_df['meta_pred_tamper'] == -1].shape[0] / attacks_df.shape[0]
    tpr = untampered_df[untampered_df['meta_pred_tamper'] == 1].shape[0] / untampered_df.shape[0]


    st.write('TNR: {} / TPR: {} / F20: {}'.format(tnr, tpr, f_beta))

def train_qoe_model(data_df):
    """
    Function to train model from given dataset
    """
    num_train = int(data_df.shape[0]*0.8)

    train_data = data_df.sample(num_train)
    test_data = data_df[~data_df.index.isin(train_data.index)]

    categorical_features_indices = []

    train_pool = Pool(data=train_data[FEATURES_QOE],
                      label=train_data[METRICS_QOE],
                      cat_features=categorical_features_indices)

    loss_funct = 'MAE'
    model_catbootregressor = CatBoostRegressor(depth=6,
                                               num_trees=1000,
                                               l2_leaf_reg=5,
                                               learning_rate=0.05,
                                               loss_function=loss_funct
                                               )
    #train the model
    print('Training QoE model:')
    model_catbootregressor.fit(train_pool)

    test_pool = Pool(data=test_data[FEATURES_QOE],
                      label=test_data[METRICS_QOE],
                      cat_features=categorical_features_indices)
    learn_train_df = pd.DataFrame(model_catbootregressor.eval_metrics(train_pool, ['MAE', 'MAPE', 'RMSE']))
    learn_test_df = pd.DataFrame(model_catbootregressor.eval_metrics(test_pool, ['MAE', 'MAPE', 'RMSE']))

    st.write('QoE model test set accuracy:')
    st.write(learn_test_df.min())

    return model_catbootregressor, test_data

def train_tamper_model(data_df):
    """
    Trains an unsupervised learning model for tamper verification
    """
    # Get untampered assets dataset
    df_1 = data_df[data_df['tamper'] == 1]
    # Get tampered (attacks) dataset
    df_0 = data_df[~data_df.index.isin(df_1.index)]

    num_train = int(df_1.shape[0] * 0.8)
    # Split dataset into train, test and attacks and shuffle them
    df_train_ul = df_1.sample(num_train)
    df_test_ul = df_1[~df_1.index.isin(df_train_ul.index)]
    df_attacks_ul = df_0.sample(frac=1)

    df_train_ul = rescale_to_resolution(df_train_ul)
    df_test_ul = rescale_to_resolution(df_test_ul)
    df_attacks_ul = rescale_to_resolution(df_attacks_ul)
    # Convert datasets from dataframes to numpy arrays
    x_train_ul = np.asarray(df_train_ul[FEATURES])
    x_test_ul = np.asarray(df_test_ul[FEATURES])
    x_attacks_ul = np.asarray(df_attacks_ul[FEATURES])

    # Scale the data
    scaler = StandardScaler()
    x_train_ul = scaler.fit_transform(x_train_ul)
    x_test_ul = scaler.transform(x_test_ul)
    x_attacks_ul = scaler.transform(x_attacks_ul)

    # Define One Class Support Vector Machine model and train it
    oc_svm = svm.OneClassSVM(kernel='rbf', gamma=0.05, nu=0.001, cache_size=5000)
    oc_svm.fit(x_train_ul)

    # Evaluate its accuracy
    f_beta, area, tnr, tpr_train, tpr_test = model_evaluation(oc_svm,
                                                              x_train_ul,
                                                              x_test_ul,
                                                              x_attacks_ul
                                                              )
    st.write('UNSUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / Area:{} / TNR:{} / TPR:{}'.format(f_beta, area, tnr, tpr_test))

    # Now use the unsupervised trained model to generate a supervised model
    # using the predictions
    num_train = int(data_df.shape[0] * 0.8)
    df_train_sl = data_df.sample(num_train)
    df_test_sl = data_df[~data_df.index.isin(df_train_sl.index)]
    df_attacks_sl = df_0.sample(frac=1)
    st.write('Attacks size:', df_attacks_sl.shape)
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('black_and_white')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('rotate')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vignette')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vertical')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('345x114')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('856x856')]

    st.write(df_train_sl['rendition'].unique())
    x_train_sl = scaler.transform(np.asarray(df_train_sl[FEATURES]))
    y_train_sl = df_train_sl['tamper']

    cat_features = []
     # Initialize CatBoostClassifier
    catboost_binary = CatBoostClassifier(iterations=500,
                                         learning_rate=0.01,
                                         depth=6)
    # Fit model
    catboost_binary.fit(np.asarray(df_train_sl[FEATURES_SL]), y_train_sl, cat_features)

    # Evaluate its accuracy
    f_beta, area, tnr, tpr_train, tpr_test = model_evaluation(catboost_binary,
                                                              df_train_sl[FEATURES_SL],
                                                              df_test_sl[FEATURES_SL],
                                                              df_attacks_sl[FEATURES_SL]
                                                              )
    st.write('SUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / Area:{} / TNR:{} / TPR_train:{} / TPR_test:{}'.format(f_beta, area, tnr, tpr_train, tpr_test))

    return catboost_binary, oc_svm, scaler, df_attacks

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

    df_aggregated['pred_ssim'] = qoe_model.predict(df_aggregated[FEATURES_QOE])
    # Train unsupervised tamper verification and extract attacks dataset
    catboost_binary, oc_svm, scaler, df_attacks = train_tamper_model(df_aggregated)

    # Evaluate incidence of false positives from attack dataset
    x_attacks = np.asarray(df_attacks[FEATURES])
    x_attacks = scaler.transform(x_attacks)
    df_attacks['ul_pred_tamper'] = oc_svm.predict(x_attacks)
    df_attacks['sl_pred_tamper'] = catboost_binary.predict(df_attacks[FEATURES])
    df_attacks['sl_pred_tamper'] = df_attacks['sl_pred_tamper'].apply(lambda x: 1 if  x == 1 else -1)

    df_attacks['meta_pred_tamper'] = df_attacks.apply(meta_model, axis=1)
    st.write('ATTACKS')
    st.write(df_attacks.head(100))
    false_positives_df = df_attacks[df_attacks['tamper'] != df_attacks['meta_pred_tamper']]
    st.write(false_positives_df.groupby('rendition').count())

    # Add predictions to test data set
    test_data_qoe['pred_ssim'] = qoe_model.predict(test_data_qoe[FEATURES_QOE])

    x_test = scaler.transform(test_data_qoe[FEATURES])
    test_data_qoe['ul_pred_tamper'] = oc_svm.predict(x_test)
    
    test_data_qoe['sl_pred_tamper'] = catboost_binary.predict(test_data_qoe[FEATURES_SL])
    test_data_qoe['sl_pred_tamper'] = test_data_qoe['sl_pred_tamper'].apply(lambda x: 1 if  x == 1 else -1)

    test_data_qoe['meta_pred_tamper'] = test_data_qoe.apply(meta_model, axis=1)
    st.write('TEST')
    st.write(test_data_qoe.head(100))

    # Display correlation between predicted and measured metric and color them
    # according to their tamper classification
    fig = px.scatter(test_data_qoe,
                     x='pred_ssim',
                     y=METRICS_QOE,
                     color='meta_pred_tamper',
                     hover_data=['rendition'])
    st.plotly_chart(fig)

    df_features = pd.DataFrame(df_qoe[FEATURES_QOE + METRICS_QOE])
    corr = df_features.corr()
    corr.style.background_gradient(cmap='coolwarm')
    fig = go.Figure(data=go.Heatmap(x=FEATURES_QOE + METRICS_QOE,
                                    y=FEATURES_QOE + METRICS_QOE,
                                    z=corr
                                    ))
 
    st.plotly_chart(fig, width=1000, height=1000)
 
    meta_model_evaluation(df_aggregated, catboost_binary, oc_svm, scaler)
if __name__ == '__main__':

    main()
