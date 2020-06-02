"""
Module wrapping up VideoAssetProcessor class in order to serve as interface for
CLI and API.
It manages pre-verification and tamper verfication of assets
"""

import uuid
import time
import json
import tarfile
import os
import sys
import urllib
import subprocess

from joblib import load
import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

sys.path.insert(0, 'scripts/asset_processor')

from video_asset_processor import VideoAssetProcessor


def pre_verify(source, rendition):
    """
    Function to verify that rendition conditions and specifications
    are met as prescribed by the Broadcaster
    """
    # Extract data from video capture
    video_file, audio_file, video_available, audio_available = retrieve_video_file(rendition['uri'])
    rendition['video_available'] = video_available
    rendition['audio_available'] = audio_available

    if video_available:
        # Check that the audio exists
        if audio_available and source['audio_available']:

            _, source_file_series = wavfile.read(source['audio_path'])
            _, rendition_file_series = wavfile.read(audio_file)

            try:
                # Compute the Euclidean distance between source's and rendition's signals
                rendition['audio_dist'] = np.linalg.norm(source_file_series-rendition_file_series)
            except:
                # Set to negative to indicate an error during audio comparison
                # (matching floating-point datatype of Euclidean distance)
                rendition['audio_dist'] = -1.0
            # Cleanup the audio file generated to avoid cluttering
            os.remove(audio_file)

        rendition_capture = cv2.VideoCapture(video_file)
        fps = int(rendition_capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(rendition_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height = float(rendition_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = float(rendition_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        rendition_copy = rendition.copy()
        rendition['path'] = video_file

        # Create dictionary with passed / failed verification parameters

        for key in rendition_copy:
            if key == 'resolution':

                rendition['resolution']['height_pre_verification'] = height / float(rendition['resolution']['height'])
                rendition['resolution']['width_pre_verification'] = width / float(rendition['resolution']['width'])

            if key == 'frame_rate':
                rendition['frame_rate'] = 0.99 <= fps / float(rendition['frame_rate']) <= 1.01

            if key == 'bitrate':
                # Compute bitrate
                duration = float(frame_count) / float(fps) # in seconds
                bitrate = os.path.getsize(video_file) / duration
                rendition['bitrate'] = bitrate == rendition['bitrate']

            if key == 'pixels':
                rendition['pixels_pre_verification'] = float(rendition['pixels']) / frame_count * height * width

    return rendition

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

def verify(source_uri, renditions, do_profiling, max_samples, model_dir, model_name_ul, model_name_sl, model_name_qoe, video_asset_processor=VideoAssetProcessor, debug=False, use_gpu=False):
    """
    Function that returns the predicted compliance of a list of renditions
    with respect to a given source file using a specified model.
    """

    total_start = time.clock()
    total_start_user = time.time()

    source_video, source_audio, video_available, audio_available = retrieve_video_file(source_uri)

    if video_available:
    # Prepare source and renditions for verification
        source = {'path': source_video,
                  'audio_path' : source_audio,
                  'video_available': video_available,
                  'audio_available': audio_available,
                  'uri': source_uri}

        # Create a list of preverified renditions
        pre_verified_renditions = []
        for rendition in renditions:
            pre_verification = pre_verify(source, rendition)
            if rendition['video_available']:
                pre_verified_renditions.append(pre_verification)

        # Cleanup the audio file generated to avoid cluttering
        if os.path.exists(source['audio_path']):
            os.remove(source['audio_path'])

        # Configure UL model for inference
        model_name_ul = 'OCSVM'
        scaler_type = 'StandardScaler'
        learning_type = 'UL'
        loaded_model_ul = load(open('{}/{}.joblib'.format(model_dir,
                                                          model_name_ul), 'rb'))

        loaded_scaler = load(open('{}/{}_{}.joblib'.format(model_dir,
                                                           learning_type,
                                                           scaler_type), 'rb'))
        # Configure SL model for inference
        model_name_sl = 'CB_Binary'
        loaded_model_sl = CatBoostClassifier().load_model('{}/{}.cbm'.format(model_dir,
                                                                             model_name_sl))

        # Configure SL model for inference
        model_name_qoe = 'CB_Regressor'
        loaded_model_qoe = CatBoostRegressor().load_model('{}/{}.cbm'.format(model_dir,
                                                                             model_name_qoe))
        # Open model configuration files
        with open('{}/param_{}.json'.format(model_dir, model_name_ul)) as json_file:
            params = json.load(json_file)
            features_ul = params['features']
        with open('{}/param_{}.json'.format(model_dir, model_name_sl)) as json_file:
            params = json.load(json_file)
            features_sl = params['features']
        with open('{}/param_{}.json'.format(model_dir, model_name_qoe)) as json_file:
            params = json.load(json_file)
            features_qoe = params['features']
        # Remove non numeric features from feature list
        non_temporal_features = ['attack_ID', 'title', 'attack', 'dimension', 'size', 'size_dimension_ratio']
        metrics_list = []
        features = list(np.unique(features_ul + features_sl + features_qoe))

        for metric in features:
            if metric not in non_temporal_features:
                metrics_list.append(metric.split('-')[0])

        # Initialize times for assets processing profiling
        start = time.clock()
        start_user = time.time()

        # Instantiate VideoAssetProcessor class
        asset_processor = video_asset_processor(source,
                                              pre_verified_renditions,
                                              metrics_list,
                                              do_profiling,
                                              max_samples,
                                              features,
												debug,
												use_gpu)

        # Record time for class initialization
        initialize_time = time.clock() - start
        initialize_time_user = time.time() - start_user

        # Register times for asset processing
        start = time.clock()
        start_user = time.time()

        # Assemble output dataframe with processed metrics
        metrics_df, pixels_df, dimensions_df = asset_processor.process()

        # Record time for processing of assets metrics
        process_time = time.clock() - start
        process_time_user = time.time() - start_user

        x_renditions_sl = np.asarray(metrics_df[features_sl])
        x_renditions_ul = np.asarray(metrics_df[features_ul])
        x_renditions_ul = loaded_scaler.transform(x_renditions_ul)
        x_renditions_qoe = np.asarray(metrics_df[features_qoe])

        np.set_printoptions(precision=6, suppress=True)
        print('INPUT SL ARRAY:', x_renditions_sl, flush=True)
        print('Unscaled INPUT UL ARRAY:', np.asarray(metrics_df[features_ul]), flush=True)
        print('SCALED INPUT UL ARRAY:', x_renditions_ul, flush=True)
        print('INPUT QOE ARRAY:', x_renditions_qoe, flush=True)
        # Make predictions for given data
        start = time.clock()
        predictions_df = pd.DataFrame()
        predictions_df['sl_pred_tamper'] = loaded_model_sl.predict(x_renditions_sl)
        predictions_df['ssim_pred'] = loaded_model_qoe.predict(x_renditions_qoe)
        predictions_df['ocsvm_dist'] = loaded_model_ul.decision_function(x_renditions_ul)
        predictions_df['ul_pred_tamper'] = loaded_model_ul.predict(x_renditions_ul)
        predictions_df['meta_pred_tamper'] = predictions_df.apply(meta_model, axis=1)
        prediction_time = time.clock() - start

        # Add predictions to rendition dictionary
        i = 0
        for _, rendition in enumerate(renditions):
            if rendition['video_available']:
                rendition.pop('path', None)
                rendition['ssim_pred'] = float(predictions_df['ssim_pred'].iloc[i])
                rendition['ocsvm_dist'] = float(predictions_df['ocsvm_dist'].iloc[i])
                rendition['tamper_ul'] = int(predictions_df['ul_pred_tamper'].iloc[i])
                rendition['tamper_sl'] = int(predictions_df['sl_pred_tamper'].iloc[i])
                rendition['tamper_meta'] = int(predictions_df['meta_pred_tamper'].iloc[i])
                # Append the post-verification of resolution and pixel count
                if 'pixels' in rendition:
                    rendition['pixels_post_verification'] = float(rendition['pixels']) / pixels_df[i]
                if 'resolution' in rendition:
                    rendition['resolution']['height_post_verification'] = float(rendition['resolution']['height']) / int(dimensions_df[i].split(':')[0])
                    rendition['resolution']['width_post_verification'] = float(rendition['resolution']['width']) / int(dimensions_df[i].split(':')[1])
                i += 1

        if do_profiling:
            print('Features used:', features)
            print('Total CPU time:', time.clock() - total_start)
            print('Total user time:', time.time() - total_start_user)
            print('Initialization CPU time:', initialize_time)
            print('Initialization user time:', initialize_time_user)

            print('Process CPU time:', process_time)
            print('Process user time:', process_time_user)
            print('Prediction CPU time:', prediction_time)

    return renditions


def retrieve_models(uri):
    """
    Function to obtain pre-trained model for verification predictions
    """

    model_dir = '/tmp/model'
    model_file = uri.split('/')[-1]
    model_file_sl = f'{model_file}_cb_sl'
    model_file_qoe = f'{model_file}_cb_qoe'
    # Create target Directory if don't exist
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print('Directory ', model_dir, ' Created ')
        print('Model download started!')
        filename, _ = urllib.request.urlretrieve(uri,
                                                 filename='{}/{}'.format(model_dir, model_file))
        print(f'Model {filename} downloaded')
        try:
            with tarfile.open(filename) as tar_f:
                tar_f.extractall(model_dir)

            return model_dir, model_file, model_file_sl, model_file_qoe
        except Exception:
            return 'Unable to untar model'
    else:
        print('Directory ', model_dir, ' already exists, skipping download')
        return model_dir, model_file, model_file_sl, model_file_qoe


def retrieve_video_file(uri):
    """
    Function to obtain a path to a video file from url or local path
    """
    video_file = ''
    audio_file = ''
    video_available = True
    audio_available = True

    if 'http' in uri:
        try:
            file_name = '/tmp/{}'.format(uuid.uuid4())

            print('File download started!', file_name, flush=True)
            video_file, _ = urllib.request.urlretrieve(uri, filename=file_name)

            print('File {} downloaded to {}'.format(file_name, video_file), flush=True)
        except Exception as e:
            print('Unable to download HTTP video file:', e, flush=True)
            video_available = False
    else:
        if os.path.isfile(uri):
            video_file = uri

            print('Video file {} available in file system'.format(video_file), flush=True)
        else:
            video_available = False
            print('File {} NOT available in file system'.format(uri), flush=True)

    if video_available:
        try:
            audio_file = '{}_audio.wav'.format(video_file)
            print('Extracting audio track')
            subprocess.call(['ffmpeg',
                         '-i',
                         video_file,
                         '-vn',
                         '-acodec',
                         'pcm_s16le',
                         '-loglevel',
                         'quiet',
                         audio_file])
        except:
            print('Could not extract audio from video file {}'.format(video_file))
            audio_available = False
        if os.path.isfile(audio_file):
            print('Audio file {} available in file system'.format(audio_file), flush=True)
        else:
            print('Audio file {} NOT available in file system'.format(audio_file), flush=True)
            audio_available = False

    return video_file, audio_file, video_available, audio_available
