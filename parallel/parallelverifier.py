
import uuid
import timeit
import json
import tarfile
import os
import sys
import urllib
import subprocess
import logging
from joblib import load
import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from verifier import file_locker

from scripts.asset_processor.video_asset_processor import VideoAssetProcessor
from parallel.parallelgraber import ParallelGraber
from parallel.parallelcompare import SampleCompare


logger = logging.getLogger()


class ParallelVerifier:
    def __init__(self, max_samples, model, use_gpu, do_profiling, debug):
        """
        Initialize verifier instance
        @param max_samples: Max number of samples to take for a video
        @param model: Either URI of the archive with model files, or local folder path
        @param use_gpu: Use GPU for video decoding and computations
        @param do_profiling: Output execution times to logs
        @param debug: Enable debug image output, greatly reduces performance
        """
        self.use_gpu = use_gpu
        self.debug = debug
        self.model_dir = '/tmp/model'
        if os.path.isdir(model):
            self.model_dir = model
        else:
            self.retrieve_models(model, self.model_dir)
        self.max_samples = max_samples
        self.do_profiling = do_profiling
        self.tmp_files = []
        self.load_models()

    @staticmethod
    def read_video_metadata(filename):
        try:
            res = {}
            cap = cv2.VideoCapture(filename)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            res['frame_rate'] = fps
            res['resolution'] = {'width': width, 'height': height}
            res['pixels'] = width * height * frame_count
            res['duration'] = frame_count / fps
            res['bitrate'] = os.path.getsize(filename) / res['duration']
        finally:
            cap.release()
        return res

    def pre_verify(self, source, rendition):
        """
        Function to verify that rendition conditions and specifications
        are met as prescribed by the Broadcaster
        """
        # Extract data from video capture
        video_file, audio_file = self.get_video_audio(rendition['uri'])
        rendition['video_available'] = video_file is not None
        rendition['audio_available'] = audio_file is not None

        if video_file:
            # Check that the audio exists
            if audio_file:
                _, source_file_series = wavfile.read(source['audio_path'])
                _, rendition_file_series = wavfile.read(audio_file)

                try:
                    # Compute the Euclidean distance between source's and rendition's signals
                    rendition['audio_dist'] = np.linalg.norm(source_file_series - rendition_file_series)
                except:
                    # Set to negative to indicate an error during audio comparison
                    # (matching floating-point datatype of Euclidean distance)
                    rendition['audio_dist'] = -1.0
                finally:
                    # Cleanup the audio file generated to avoid cluttering
                    os.remove(audio_file)

            metadata = self.read_video_metadata(video_file)

            rendition['path'] = video_file

            # Create dictionary with passed / failed verification parameters
            if rendition.get('resolution'):
                rendition['resolution']['height_pre_verification'] = metadata['height'] / float(rendition['resolution']['height'])
                rendition['resolution']['width_pre_verification'] = metadata['width'] / float(rendition['resolution']['width'])

            if 'frame_rate' in rendition:
                rend_exp_fps = float(rendition['frame_rate']) or source['frame_rate']
                rendition['frame_rate'] = bool(np.isclose(rend_exp_fps, metadata['frame_rate'], atol=0.1))

            if rendition.get('bitrate'):
                rendition['bitrate'] = metadata['bitrate'] == rendition['bitrate']

            if rendition.get('pixels'):
                rendition['pixels_pre_verification'] = float(rendition['pixels']) / metadata['pixels']


        return rendition

    def meta_model(self, row):
        """
        The goal is to reduce the number of False Positives (tamper) to prevent wrongfully penalizing transcoder nodes. OCSVM model is expected to have higher precision (low FP) on novel data.
        If OCSVM predicts the observation is an inlier (not tampered), we'll go with it, otherwise we'll use supervised model output.
        """
        if row['ul_pred_tamper'] == 0:
            return 0
        return row['sl_pred_tamper']

    def verify(self, source_uri, renditions):
        """
        Function that returns the predicted compliance of a list of renditions
        with respect to a given source file using a specified model.
        """
        total_start = timeit.default_timer()
        source_video, source_audio = self.get_video_audio(source_uri)
        if not source_video and not source_audio:
            raise ValueError('Couldn\'t retrieve source files')
        try:
            if source_video:
                # Prepare source and renditions for verification
                source = {'path': source_video,
                          'audio_path': source_audio,
                          'video_available': True,
                          'audio_available': source_audio is not None,
                          'uri': source_uri}
                # read source metadata
                metadata = self.read_video_metadata(source_video)
                source.update(metadata)

                # Create a list of preverified renditions
                pre_verified_renditions = []
                for rendition in renditions:
                    pre_verification = self.pre_verify(source, rendition)
                    if rendition['video_available']:
                        pre_verified_renditions.append(pre_verification)

                # Remove non numeric features from feature list
                non_temporal_features = ['attack_ID', 'title', 'attack', 'dimension', 'size', 'size_dimension_ratio']
                metrics_list = []
                features = list(np.unique(self.features_ul + self.features_sl))

                for metric in features:
                    if metric not in non_temporal_features:
                        metrics_list.append(metric.split('-')[0])

                #----- Frame grabber -----
                # Frame grabber times
                start = timeit.default_timer()

                # Instantiate Frame Grabber class
                samplegraber = ParallelGraber(self.max_samples, self.use_gpu, self.do_profiling, self.debug)

                samplegraber.addgraber(source_video)
                for rendition in renditions:
                    samplegraber.addgraber(rendition['path'])
                sampledata = samplegraber.captureall()

                initialize_time = timeit.default_timer() - start

                # ----- Feature calculation -----
                # Register times for asset processing
                start = timeit.default_timer()

                featureCompare = SampleCompare(metrics_list, features, self.do_profiling)

                # Assemble output dataframe with processed metrics
                metrics_df, pixels_df, dimensions_df = featureCompare.process(sampledata, renditions, source_video)

                # Record time for processing of assets metrics
                process_time = timeit.default_timer() - start

                # ----- Inference part -----
                x_renditions_sl = np.asarray(metrics_df[self.features_sl])
                x_renditions_ul = np.asarray(metrics_df[self.features_ul])
                x_renditions_ul = self.loaded_scaler.transform(x_renditions_ul)

                np.set_printoptions(precision=6, suppress=True)
                logger.debug(f'INPUT SL ARRAY: {x_renditions_sl}')
                logger.debug(f'Unscaled INPUT UL ARRAY: {np.asarray(metrics_df[self.features_ul])}')
                logger.debug(f'SCALED INPUT UL ARRAY: {x_renditions_ul}')
                # Make predictions for given data
                start = timeit.default_timer()
                predictions_df = pd.DataFrame()
                predictions_df['sl_pred_tamper'] = self.loaded_model_sl.predict(x_renditions_sl)
                predictions_df['ocsvm_dist'] = self.loaded_model_ul.decision_function(x_renditions_ul)
                predictions_df['ul_pred_tamper'] = (-self.loaded_model_ul.predict(x_renditions_ul)+1)/2
                predictions_df['meta_pred_tamper'] = predictions_df.apply(self.meta_model, axis=1)
                prediction_time = timeit.default_timer() - start

                # Add predictions to rendition dictionary
                i = 0
                for _, rendition in enumerate(renditions):
                    if rendition['video_available']:
                        rendition.pop('path', None)
                        rendition['ocsvm_dist'] = float(predictions_df['ocsvm_dist'].iloc[i])
                        rendition['tamper_ul'] = int(predictions_df['ul_pred_tamper'].iloc[i])
                        rendition['tamper_sl'] = int(predictions_df['sl_pred_tamper'].iloc[i])
                        rendition['tamper'] = int(predictions_df['meta_pred_tamper'].iloc[i])
                        # Append the post-verification of resolution and pixel count
                        if 'pixels' in rendition:
                            rendition['pixels_post_verification'] = float(rendition['pixels']) / pixels_df[i]
                        if 'resolution' in rendition:
                            rendition['resolution']['height_post_verification'] = float(rendition['resolution']['height']) / int(dimensions_df[i].split(':')[0])
                            rendition['resolution']['width_post_verification'] = float(rendition['resolution']['width']) / int(dimensions_df[i].split(':')[1])
                        i += 1

                if self.do_profiling:
                    logger.info(f'Features used: {features}')
                    logger.info(f'Total time: {timeit.default_timer() - total_start}')
                    logger.info(f'Initialization time: {initialize_time}')
                    logger.info(f'Process time: {process_time}')
                    logger.info(f'Prediction time: {prediction_time}')

            return renditions
        finally:
            for f in self.tmp_files:
                if os.path.exists(f):
                    os.remove(f)
            self.tmp_files.clear()

    def retrieve_models(self, uri, model_dir):
        """
        Function to obtain pre-trained model for verification predictions
        """
        with file_locker.FileLocker('model_op.lock'):
            model_file = uri.split('/')[-1]
            model_file_sl = f'{model_file}_cb_sl'
            # Create target Directory if don't exist
            if not os.path.exists(model_dir):
                try:
                    os.mkdir(model_dir)
                    logger.info(f'Directory created: {model_dir}')
                    logger.info('Model download started')
                    filename, _ = urllib.request.urlretrieve(uri,
                                                             filename='{}/{}'.format(model_dir, model_file))
                    logger.info(f'Model {filename} downloaded')
                    with tarfile.open(filename) as tar_f:
                        tar_f.extractall(model_dir)

                    return model_dir, model_file, model_file_sl
                except Exception as exc:
                    if os.path.exists(model_dir):
                        os.rmdir(model_dir)
                    logger.exception('Unable to untar model')
                    raise exc
            else:
                logger.debug(f'Directory {model_dir} already exists, skipping download')

    def get_video_audio(self, uri):
        """
        Function to obtain a path to a video and audio files from url or local path
        """
        video_file = None
        audio_file = None
        if uri.lower().startswith('http'):
            try:
                file_name = '/tmp/{}'.format(uuid.uuid4())
                logger.info(f'File download started: {file_name}')
                video_file, _ = urllib.request.urlretrieve(uri, filename=file_name)
                self.tmp_files.append(video_file)
                logger.info(f'File {file_name} downloaded to {video_file}')
            except Exception as e:
                logger.exception('Unable to download HTTP video file')
        else:
            if os.path.isfile(uri):
                video_file = uri
                logger.info(f'Video file {video_file} available in file system')
            else:
                logger.info(f'Video file {video_file} NOT available in file system')

        if video_file:
            audio_file = '{}_audio.wav'.format(video_file)
            logger.info('Extracting audio track')
            ffmpeg = subprocess.Popen(' '.join(['ffmpeg',
                             '-i',
                             video_file,
                             '-vn',
                             '-acodec',
                             'pcm_s16le',
                             '-loglevel',
                             'quiet',
                             audio_file]), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
            stdout, stderr = ffmpeg.communicate()
            if ffmpeg.returncode:
                logger.error(f'Could not extract audio from video file {stderr}')
            if os.path.isfile(audio_file):
                logger.info(f'Audio file {audio_file} available in file system')
                self.tmp_files.append(audio_file)
            else:
                logger.info(f'Audio file {audio_file} NOT available in file system')
                audio_file = None
        return video_file, audio_file

    def load_models(self):
        """
        Cache models to memory
        @return:
        """
        # Configure UL model for inference
        model_name_ul = 'OCSVM'
        scaler_type = 'StandardScaler'
        learning_type = 'UL'
        self.loaded_model_ul = load(open('{}/{}.joblib'.format(self.model_dir,
                                                               model_name_ul), 'rb'))

        self.loaded_scaler = load(open('{}/{}_{}.joblib'.format(self.model_dir,
                                                                learning_type,
                                                                scaler_type), 'rb'))
        # Configure SL model for inference
        model_name_sl = 'CB_Binary'
        self.loaded_model_sl = CatBoostClassifier().load_model('{}/{}.cbm'.format(self.model_dir,
                                                                                  model_name_sl))

        # Open model configuration files
        with open('{}/param_{}.json'.format(self.model_dir, model_name_ul)) as json_file:
            params = json.load(json_file)
            self.features_ul = params['features']
        with open('{}/param_{}.json'.format(self.model_dir, model_name_sl)) as json_file:
            params = json.load(json_file)
            self.features_sl = params['features']