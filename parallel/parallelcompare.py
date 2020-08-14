import os
import shutil
import timeit

import cv2
import numpy as np
import pandas as pd
import logging


from parallel.parallelgraber import Sample

from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import multiprocessing
from scipy.spatial import distance
from scripts.asset_processor.video_metrics import VideoMetrics

from scripts.asset_processor.video_asset_processor import VideoAssetProcessor
from scripts.asset_processor.video_capture import VideoCapture

class SampleCompare:
    def __init__(self, metrics_list, features_list, do_profiling=False):
        """
        @param use_gpu:
        @param original:
        @param renditions:
        @param metrics_list:
        @param do_profiling:
        @param max_samples: Max number of matched master-rendition frames to calculate metrics against. -1 = all
        @param features_list:
        @param debug_frames: dump frames selected for metric extraction on disk, decreases performance
        @param channel: which HSV channel (0-3) to use for metric computation, -1 = all
        @param image_pair_callback: function to call when image pair is created
        """
        # ************************************************************************
        # Initialize global variables
        # ************************************************************************
        self.hash_size = 16
        self.metrics_list = metrics_list
        self.height = 0
        self.features_list = features_list

        self.metrics = {}

        if do_profiling:
            import line_profiler
            self.cpu_profiler = line_profiler.LineProfiler()
        else:
            self.cpu_profiler = None
        self.do_profiling = do_profiling

        # Check if HD list is necessary
        if 'temporal_ssim' in self.metrics_list or 'temporal_psnr' in self.metrics_list:
            self.make_hd_list = True
        else:
            self.make_hd_list = False

        self.video_metrics = VideoMetrics(self.metrics_list,
                                          self.hash_size,
                                          int(self.height),
                                          self.cpu_profiler,
                                          self.do_profiling)

    def process(self, sampledata, renditions_list, source_video):

        self.renditions_list = renditions_list
        self.original_path = source_video
        self.fps = sampledata[0].fps

        original_metrics = {}
        for i in range(len(sampledata[0].samples) - 1):
            key = f'{i}'
            result_metrics = {}
            for metric in self.metrics_list:
                result_metrics[metric] = 0.0
            result_metrics['dimensions'] = '{}:{}'.format(int(sampledata[0].width), int(sampledata[0].height))
            result_metrics['pixels'] = sampledata[0].pixels
            result_metrics['ID'] = self.original_path
            original_metrics[key] = result_metrics

        self.metrics[self.original_path] = original_metrics

        i = -1
        for rendition in renditions_list:
            path = rendition['path']
            i += 1
            try:
                if os.path.exists(path):
                    # Compute the metrics for the rendition
                    self.metrics[path] = self.compute(sampledata[0], sampledata[i+1], path)
                else:
                    #logger.error(f'Unable to find rendition file: {path}')
                    print(f'Unable to find rendition file: {path}')
            except Exception as err:
                #logger.exception('Unable to compute metrics for {}'.format(path))
                print('Unable to compute metrics for {}'.format(path))
            finally:
                print('Finish to compute metrics for {}'.format(path))

        return self.aggregate(self.metrics)

    def aggregate(self, metrics):
        """
        Function to aggregate computed values of metrics and renditions into a
        pandas DataFrame.
        """

        # Dictionary for containing all metrics
        metrics_dict = {}
        # Dictionary for containing all renditions
        renditions_dict = {}

        # Aggregate dictionary with all values for all renditions into a Pandas DataFrame
        # All values are stored and obtained in a per-frame basis, then in a per-rendition
        # fashion. They need to be rearranged.

        # First, we combine the frames
        dict_of_df = {k: pd.DataFrame(v) for k, v in metrics.items()}
        metrics_df = pd.concat(dict_of_df, axis=1, sort=True).transpose().reset_index(inplace=False)

        # Pandas concat function creates a level_0 and level_1 extra columns.
        # They need to be renamed
        metrics_df = metrics_df.rename(index=str,
                                       columns={"level_1": "frame_num", "level_0": "path"})

        # Then we can combine each rendition
        for rendition in self.renditions_list:
            # For the current rendition, we need an empty dictionary
            rendition_dict = {}

            # We have a number of different metrics that have been computed.
            # These are an input for the constructor of the class an vary according to
            # what metrics are of interest in the research
            for metric in self.metrics_list:
                # Obtain a Pandas DataFrame from the original and build the original time series
                original_df = metrics_df[metrics_df['path'] == self.original_path][metric]
                original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)
                # Obtain a Pandas DataFrame from the current rendition and build its time series
                rendition_df = metrics_df[metrics_df['path'] == rendition['path']][metric]
                rendition_df = rendition_df.reset_index(drop=True)
                rendition_df = rendition_df.transpose().dropna().astype(float)

                # For those metrics that have a temporal character,
                # we need to make a further aggregation
                # We are basically using the Manhattan and euclidean distances,
                # and statistically meaningful
                # values such as mean, max and standard deviation.
                # The whole time series is also provided for later exploration
                #  in the analysis part.
                if 'temporal' in metric:
                    x_original = np.array(original_df[rendition_df.index].values)
                    x_rendition = np.array(rendition_df.values)

                    [[manhattan]] = distance.cdist(x_original.reshape(1, -1),
                                                   x_rendition.reshape(1, -1),
                                                   metric='cityblock')

                    rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original,
                                                                                       x_rendition)
                    rendition_dict['{}-manhattan'.format(metric)] = manhattan
                    rendition_dict['{}-mean'.format(metric)] = np.mean(x_rendition)
                    rendition_dict['{}-max'.format(metric)] = np.max(x_rendition)
                    rendition_dict['{}-std'.format(metric)] = np.std(x_rendition)
                    rendition_dict['{}-corr'.format(metric)] = np.correlate(x_original,
                                                                            x_rendition,
                                                                            mode='same').mean()
                    rendition_dict['{}-series'.format(metric)] = x_rendition

                # Other metrics do not need time evaluation
                else:
                    rendition_dict[metric] = rendition_df.mean()

            # Size is an important feature of an asset, as it gives important information
            # regarding the potential compression effect
            rendition_dict['size'] = os.path.getsize(rendition['path'])
            rendition_dict['fps'] = self.fps
            rendition_dict['path'] = rendition['path']

            # Extract the dimensions of the rendition
            dimensions_df = metrics_df[metrics_df['path'] == rendition['path']]['dimensions']
            rendition_dict['dimension_x'] = int(dimensions_df.unique()[0].split(':')[1])
            rendition_dict['dimension_y'] = int(dimensions_df.unique()[0].split(':')[0])

            # Extract the pixels for this rendition
            pixels_df = metrics_df[metrics_df['path'] == rendition['path']]['pixels']
            rendition_dict['pixels'] = int(pixels_df.unique())

            # Store the rendition values in the dictionary of renditions for the present asset
            renditions_dict[rendition['path']] = rendition_dict

        # Add the current asset values to the global metrics_dict
        metrics_dict[self.original_path] = renditions_dict

        dict_of_df = {k: pd.DataFrame(v) for k, v in metrics_dict.items()}
        metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)

        pixels_df = metrics_df['pixels']

        # Compute a size/dimension ratio column for better accuracy
        metrics_df['size_dimension_ratio'] = metrics_df['size'] / (metrics_df['dimension_y'] * metrics_df['dimension_x'])

        metrics_df = self.cleanup_dataframe(metrics_df, self.features_list)

        return metrics_df, pixels_df, dimensions_df

    def cleanup_dataframe(self, metrics_df, features):
        """
        Cleanup the resulting pandas dataframe and convert it to a numpy array
        to pass to the prediction model
        """

        metrics_df = metrics_df.rename(columns={'level_0': 'title', 'level_1': 'attack'})

        if features is not None:
            if 'attack_ID' in features:
                features.remove('attack_ID')
            # Filter out features from metrics dataframe

            # Scale measured metrics according to their resolution for better accuracy
            metrics_df = self.rescale_to_resolution(metrics_df, features)
            metrics_df = metrics_df[features]

        return metrics_df

    @staticmethod
    def rescale_to_resolution(data, features):
        """
        Function to rescale features to improve accuracy
        """

        df_features = pd.DataFrame(data)
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
                    print('Downscaling', label, feature)
                    df_features[feature] = df_features[feature] / (df_features['dimension_y'] * df_features['dimension_x'])

        for label in upscale_features:
            upscale_feature = [feature for feature in features if label in feature]
            if upscale_feature:
                for feature in upscale_feature:
                    print('Upscaling', label, feature)
                    df_features[feature] = df_features[feature] * df_features['dimension_y'] * df_features['dimension_x']

        return df_features

    def compare_renditions_instant(self, idx, mastersample, renditionsample, path):
        """
        Function to compare pairs of numpy arrays extracting their corresponding metrics.
        It basically takes the global original frame at frame_pos and its subsequent to
        compare them against the corresponding ones in frame_list (a rendition).
        It then extracts the metrics defined in the constructor under the metrics_list.
        Methods of comparison are implemented in the video_metrics class
        @param master_sample_idx_map: Mapping from rendition sample index to master sample index. If Nframes is different between master and rendition, the index mapping is not 1:1
        @param rendition_sample_idx: Index of master sample we compare rendition against
        @param frame_list:
        @param frame_list_hd:
        @param dimensions:
        @param pixels:
        @param path:
        @return:
        """

        # Dictionary of metrics
        frame_metrics = {}
        # Original frame to compare against (downscaled for performance)
        reference_frame = mastersample.samples[idx]
        # Original's subsequent frame (downscaled for performance)
        next_reference_frame = mastersample.samples[idx+1]
        # Rendition frame (downscaled for performance)
        rendition_frame = renditionsample.samples[idx]
        # Rendition's subsequent frame (downscaled for performance)
        next_rendition_frame = renditionsample.samples[idx + 1]

        '''
        if self.debug_frames:
            cv2.imwrite(f'{self.frame_dir_name}/CRI_{idx:04}_ref.png', self._convert_debug_frame(reference_frame))
            cv2.imwrite(f'{self.frame_dir_name}/CRI_{idx:04}_next_ref.png', self._convert_debug_frame(next_reference_frame))
            cv2.imwrite(f'{self.frame_dir_name}/CRI_{idx:04}_rend.png', self._convert_debug_frame(rendition_frame))
            cv2.imwrite(f'{self.frame_dir_name}/CRI_{idx:04}_next_rend.png', self._convert_debug_frame(next_rendition_frame))
        '''

        if self.make_hd_list:
            # Original frame to compare against (HD for QoE metrics)
            reference_frame_hd = mastersample.samples_hd[idx]
            # Rendition frame (HD for QoE metrics)
            rendition_frame_hd = renditionsample.samples_hd[idx]

            # Compute the metrics defined in the global metrics_list.
            # Uses the global instance of video_metrics
            # Some metrics use a frame-to-frame comparison,
            # but other require current and forward frames to extract
            # their comparative values.
            rendition_metrics = self.video_metrics.compute_metrics(rendition_frame,
                                                                   next_rendition_frame,
                                                                   reference_frame,
                                                                   next_reference_frame,
                                                                   rendition_frame_hd,
                                                                   reference_frame_hd)
        else:
            rendition_metrics = self.video_metrics.compute_metrics(rendition_frame,
                                                                   next_rendition_frame,
                                                                   reference_frame,
                                                                   next_reference_frame)

        print(rendition_metrics)
        # Retrieve rendition dimensions for further evaluation
        dimensions = '{}:{}'.format(int(renditionsample.width), int(renditionsample.height))
        rendition_metrics['dimensions'] = dimensions

        # Retrieve rendition number of pixels for further verification
        rendition_metrics['pixels'] = renditionsample.pixels

        # Retrieve rendition path for further identification
        #rendition_metrics['ID'] = self.original_path
        rendition_metrics['ID'] = "original_path"

        # Identify rendition uniquely by its path and store metric data in frame_metrics dict
        frame_metrics[path] = rendition_metrics

        # Return the metrics, together with the position of the frame
        # frame_pos is needed for the ThreadPoolExecutor optimizations
        return rendition_metrics, idx

    def compute(self, mastersample, renditionsample, path):
        """
        Function to compare lists of numpy arrays extracting their corresponding metrics.
        It basically takes the global original list of frames and the input frame_list
        of numpy arrrays to extract the metrics defined in the constructor.
        frame_pos establishes the index of the frames to be compared.
        It is optimized by means of the ThreadPoolExecutor of Python's concurrent package
        for better parallel performance.
        @param master_sample_idx_map: Mapping from rendition sample index to master sample index. If Nframes is different between master and rendition, the index mapping is not 1:1
        @param frame_list:
        @param frame_list_hd:
        @param path:
        @param dimensions:
        @param pixels:
        @return:
        """
        start = timeit.default_timer()

        # Dictionary of metrics
        rendition_metrics = {}
        future_list = []

        # Execute computations in parallel using as many processors as possible
        # future_list is a dictionary storing all computed values from each thread
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Compare the original asset against its renditions
            for i in range(len(mastersample.samples) - 1):
                key = f'{i}'
                future = executor.submit(self.compare_renditions_instant,
                                         i,
                                         mastersample,
                                         renditionsample,
                                         path)
                
                future_list.append((key, future))

        # Once all frames in frame_list have been iterated, we can retrieve their values
        for key, future in future_list:
            # Values are retrieved in a dict, as a result of the executor's process
            result_rendition_metrics, frame_pos = future.result()
            # The computed values at a given frame
            rendition_metrics[key] = result_rendition_metrics

        time_spent = timeit.default_timer() - start
        #logger.info(f'Metrics compute took: {time_spent}')
        #print(f'Metrics compute took: {time_spent}')
        '''
        for i in range(len(mastersample.samples) - 1):
            key = f'{i}'
            result_rendition_metrics, frame_pos = self.compare_renditions_instant( i, mastersample, renditionsample, path)
            # The computed values at a given frame
            rendition_metrics[key] = result_rendition_metrics
        '''

        # Return the metrics for the currently processed rendition
        return rendition_metrics