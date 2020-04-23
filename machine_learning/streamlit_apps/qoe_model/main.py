"""
Module to generate an interactive app to visualize and train a QoE predictive model
It relies of Streamlite library for the visualization and display of widgets
"""

import os.path
import json

from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from joblib import dump
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

FEATURES = [
               'size_dimension_ratio',
               'temporal_dct-mean',
               'temporal_gaussian_mse-mean',
               'temporal_gaussian_difference-mean',
               'temporal_threshold_gaussian_difference-mean',

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
    df_data = pd.DataFrame(data)

    features = list(df_data.columns)
    downscale_features = ['temporal_psnr',
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
                if df_data[feature].dtype == int or df_data[feature].dtype == float:
                    print('Downscaling', label, feature)
                    df_data[feature] = df_data[feature] / df_data['size_dimension_ratio']

    for label in upscale_features:
        upscale_feature = [feature for feature in features if label in feature]
        if upscale_feature:
            for feature in upscale_feature:
                if df_data[feature].dtype == int or df_data[feature].dtype == float:
                    print('Upscaling', label, feature)
                    df_data[feature] = df_data[feature] * df_data['size_dimension_ratio']

    return df_data

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
    st.write(attacks_df[attacks_df['ul_pred_tamper'] == 1].groupby('rendition').count(), 'attacks:', attacks_df.shape)
    st.subheader('Supervised model false positives')
    st.write(attacks_df[attacks_df['sl_pred_tamper'] == 1].groupby('rendition').count(), attacks_df.shape)
    st.subheader('Meta model false positives')
    st.write(attacks_df[attacks_df['meta_pred_tamper'] == 1].groupby('rendition').count(), attacks_df.shape)

    st.subheader('Unsupervised model false negatives')
    st.write(untampered_df[untampered_df['ul_pred_tamper'] == -1].groupby('rendition').count(), 'untampered:', untampered_df.shape)
    st.subheader('Supervised model false negatives')
    st.write(untampered_df[untampered_df['sl_pred_tamper'] == -1].groupby('rendition').count(), untampered_df.shape)
    st.subheader('Meta model false negatives')
    st.write(untampered_df[untampered_df['meta_pred_tamper'] == -1].groupby('rendition').count(), untampered_df.shape)

def meta_model(row):
    return row['ul_pred_tamper'] if row['ul_pred_tamper'] == 1 and not row['sl_pred_tamper'] == -1 else row['sl_pred_tamper']

def train_qoe_model(data_df):
    """
    Function to train model from given dataset
    """
    num_train = int(data_df.shape[0] * 0.8)

    train_data = data_df.sample(num_train)
    test_data = data_df[~data_df.index.isin(train_data.index)]

    train_data = rescale_to_resolution(train_data)
    test_data = rescale_to_resolution(test_data)

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
    learn_test_df = pd.DataFrame(model_catbootregressor.eval_metrics(test_pool, ['MAE', 'MAPE', 'RMSE']))

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

def train_ul_tamper_model(data_df):
    """
    Trains an unsupervised learning model for tamper verification
    """
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
    oc_svm = svm.OneClassSVM(kernel='rbf', gamma=0.05, nu=0.001, cache_size=5000)
    oc_svm.fit(x_train_ul)

    # Evaluate its accuracy
    f_beta, tnr, tpr_test = ul_model_evaluation(oc_svm,
                                                x_test_ul,
                                                x_attacks
                                                              )

    # Save the scaler for inference
    dump(scaler, 'UL_StandardScaler.joblib')
    # Save the OC-SVM for inference
    dump(oc_svm, 'OCSVM.joblib')
    svm_params = oc_svm.get_params()
    svm_params['features'] = FEATURES
    svm_params['f_beta'] = f_beta
    svm_params['tnr'] = tnr
    svm_params['tpr_test'] = tpr_test
    with open('param_OCSVM.json', 'w') as fp:
        json.dump(svm_params, fp)    


    st.write('UNSUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / TNR:{} / TPR_test:{}'.format(f_beta,
                                                    tnr,
                                                    tpr_test))

    return oc_svm, scaler

def train_sl_tamper_model(data_df):
    # Now use the unsupervised trained model to generate a supervised model
    # using the predictions

    # Get untampered assets dataset
    st.write('Total samples:', data_df.shape)

    # Get tampered (attacks) dataset
    # Split dataset into train, test and attacks and shuffle them
    df_train_sl = data_df.sample(frac=0.8)
    df_test_sl = data_df[~data_df.index.isin(df_train_sl.index)]
    st.write('SL TAMPER', df_test_sl.head(100))

    st.write('SL TAMPER RESCALED', df_test_sl.head(100))
    st.write('Train / test split:', df_train_sl.shape, df_test_sl.shape)

    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('black_and_white')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('rotate')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vignette')]
    df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('vertical')]
    # df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('345x114')]
    # df_train_sl = df_train_sl.loc[~df_train_sl['rendition'].str.contains('856x856')]

    y_train_sl = df_train_sl['tamper']

    # df_train_sl['ul_pred_tamper'] = df_train_sl['ul_pred_tamper'].astype(str)
    cat_features = []
    # Initialize CatBoostClassifier
    CB_params = dict(iterations=500,
                     learning_rate=0.05,
                     depth=6)
    catboost_binary = CatBoostClassifier(**CB_params)
    # Fit model
    catboost_binary.fit(np.asarray(df_train_sl[FEATURES]), y_train_sl, cat_features)

    # Evaluate its accuracy
    f_beta, tnr, tpr_test = sl_model_evaluation(catboost_binary, df_test_sl)
    st.write('SUPERVISED TAMPER MODEL ACCURACY')
    st.write('F20:{} / TNR:{} / TPR_test:{}'.format(f_beta,
                                                    tnr,
                                                    tpr_test))

    catboost_binary.save_model('CB_Binary.cbm',
           format="cbm",
           export_parameters=None,
           pool=None)
 
    CB_params['eval_metrics'] = {'f_beta':f_beta, 'tnr':tnr, 'tpr_test':tpr_test}
    CB_params['features'] = FEATURES
    with open('param_CB_Binary.json', 'w') as outfile:
        json.dump(CB_params, outfile)

    return catboost_binary

def plot_3D(title, z_metric, z_axis, color_metric, df_aggregated):
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
                      scene = dict(xaxis_title="Vertical Resolution",
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
    df_aggregated = pd.concat(frames, ignore_index=True)
    df_aggregated = df_aggregated.sample(frac=1)
    # Display datasets
    st.subheader('Raw QoE data')
    st.write(df_qoe[FEATURES].describe(), df_qoe.shape)
    st.subheader('Raw tamper verification data')
    st.write(df_tamper[FEATURES].describe(), df_tamper.shape)
    st.subheader('Aggregated')
    st.write(df_aggregated[FEATURES].describe(), df_aggregated.shape)

    st.write('Untampered:', df_aggregated[df_aggregated['tamper'] == 1].shape)
    st.write('Tampered:', df_aggregated[df_aggregated['tamper'] == -1].shape)

    qoe_columns = FEATURES + METRICS_QOE
    ul_columns = FEATURES + ['tamper']
    sl_columns = FEATURES + ['tamper', 'rendition']
    # Train SSIM predictor and add predictions to aggregated dataframe
    qoe_model = train_qoe_model(df_qoe[qoe_columns])
    df_aggregated = rescale_to_resolution(df_aggregated)
    df_aggregated['pred_ssim'] = qoe_model.predict(df_aggregated[FEATURES])
    # Output the error check to the frontend
    df_ssim = df_aggregated[['pred_ssim', 'temporal_ssim-mean']].dropna(axis='rows')
    st.write('RMSE:', ((df_ssim['pred_ssim'] - df_ssim['temporal_ssim-mean']) ** 2).mean() ** .5)
    st.write('MAPE:', (np.abs(df_ssim['temporal_ssim-mean'] - df_ssim['pred_ssim']) / df_ssim['temporal_ssim-mean']).mean())
    
    # Train unsupervised tamper verification and get its feature scaler
    oc_svm, scaler = train_ul_tamper_model(df_aggregated[ul_columns])
    # Add predictions to data set
    x_test = scaler.transform(df_aggregated[FEATURES])
    df_aggregated['ocsvm_dist'] = oc_svm.decision_function(x_test)
    df_aggregated['ul_pred_tamper'] = oc_svm.predict(x_test)

    # Train supervised tamper verification and adds its predictions
    catboost_binary = train_sl_tamper_model(df_aggregated[sl_columns])
    
    df_aggregated['sl_pred_tamper'] = catboost_binary.predict(df_aggregated[FEATURES])
    df_aggregated['sl_pred_tamper'] = df_aggregated['sl_pred_tamper'].apply(lambda x: 1 if x == 1 else -1)

    # Apply meta-model to aggregated dataset
    df_aggregated['meta_pred_tamper'] = df_aggregated.apply(meta_model, axis=1)
    st.write('FINAL DF:', df_aggregated[FEATURES + ['rendition', 'sl_pred_tamper', 'ul_pred_tamper', 'meta_pred_tamper']].head(100))
    # Display evaluation metrics for the test dataset
    meta_model_evaluation(df_aggregated)

    df_test = pd.read_csv('test.csv')
    st.write(catboost_binary.predict(df_test[FEATURES]))
    x_test = scaler.transform(df_test[FEATURES])

    st.write(oc_svm.predict(x_test))
    df_plots_aggregated = df_aggregated.sample(n=5000, random_state=1)
    df_plots_qoe = df_aggregated[df_aggregated['tamper'] == 1].sample(n=5000, random_state=1)
    # Display correlation between measured QoE metric, resolution and size.
    plot_3D('SSIM Classifier', 'temporal_ssim-mean', 'SSIM', 'tamper', df_plots_aggregated)
    # Display correlation between measured distance to decision function, resolution and size.
    plot_3D('OC-SVM Classifier', 'ocsvm_dist', 'Distance to Decision Function', 'tamper', df_plots_aggregated)
    # Display histogram of non-tampered assets
    plot_histogram('ocsvm_dist', 'Distance to decision function', df_aggregated)
    # Display correlation between predicted and measured metric and color them
    # according to their tamper classification
    plot_scatter('SSIM vs Distance to Decision Function',
                 'ocsvm_dist',
                 'temporal_ssim-mean',
                 'Distance to Decision Function',
                 'SSIM',
                 df_plots_aggregated)
    # Display difference between predicted ssim and measured ssim
    # according to their tamper classification
    plot_scatter('Measured SSIM vs Predicted SSIM',
                 'pred_ssim',
                 'temporal_ssim-mean',
                 'Predicted SSIM',
                 'SSIM',
                 df_plots_aggregated,
                 line=True)
    # Display correlation matrix
    plot_correlation_matrix(df_plots_qoe)

if __name__ == '__main__':

    main()
