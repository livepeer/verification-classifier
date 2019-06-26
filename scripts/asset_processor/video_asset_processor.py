import cv2
import numpy as np
import pandas as pd
import os
from video_metrics import video_metrics
from concurrent.futures.thread import ThreadPoolExecutor
from scipy.spatial import distance
from memory_profiler import profile as mem_profiler
import line_profiler


class video_asset_processor:
    # Class to extract and aggregate values from video sequences.
    # It is instantiated as part of the data creation as well
    # as in the inference, both in the CLI as in the notebooks
    def __init__(self, original_path, renditions_paths, metrics_list, duration, do_profiling):
        # ************************************************************************
        # Initialize global variables
        # ************************************************************************

        self.original_path = original_path                                                      # Stores system path to original asset
        self.original = cv2.VideoCapture(self.original_path)                                    # Initializes original asset to OpenCV VideoCapture class
        self.fps = int(self.original.get(cv2.CAP_PROP_FPS))                                     # Frames Per Second of the original asset
        self.asset_length = int(self.original.get(cv2.CAP_PROP_FRAME_COUNT))                    # Counts number of frames of the asset
        self.duration = duration                                                                # Establishes how many seconds of the original asset are used
        self.skip_frames = 1                                                                    # Defines whether to use all frames or leap frog skip_frames frames
        self.hash_size = 16                                                                     # Size of the hash for frame hash analysis in video_metrics
        self.renditions = {}                                                                    # Dictionary containing dict of renditions ('frame_list',
                                                                                                #                                           'dimensions',
                                                                                                #                                           'ID')
        self.metrics = {}                                                                       # Dictionary containing dict of metrics
        self.metrics_list = metrics_list                                                        # List of metrics to be extracted from the asset and its renditions
        self.renditions_paths = renditions_paths                                                # List of paths to renditions

        # Retrieve original rendition dimensions
        self.height = self.original.get(cv2.CAP_PROP_FRAME_HEIGHT)                              # Obtains vertical dimension of the frames of the original
        self.width = self.original.get(cv2.CAP_PROP_FRAME_WIDTH)                                # Obtains horizontal dimension of the frames of the original
        self.dimensions = '{}:{}'.format(int(self.width), int(self.height))                     # Collects both dimensional values in a string
        self.cpu_profiler = line_profiler.LineProfiler()
        self.do_profiling = do_profiling
        self.video_metrics = video_metrics(self.metrics_list,
                                           self.skip_frames,
                                           self.hash_size,
                                           int(self.dimensions[self.dimensions.find(':') + 1:]),
                                           self.cpu_profiler,
                                           self.do_profiling)                                   # Instance of the video_metrics class

        # Convert OpenCV video captures of original to list
        # of numpy arrays for better performance of numerical computations
        self.original = self.capture_to_array(self.original)
        # Compute its features
        self.metrics[self.original_path] = self.compute(self.original, self.original_path, self.dimensions)
        # Store the value in the renditions dictionary
        self.renditions['original'] = {'frame_list': self.original,
                                       'dimensions': self.dimensions,
                                       'ID': self.original_path.split('/')[-2]}

    def capture_to_array(self, capture):
        # ************************************************************************
        # Function to convert OpenCV video capture to a list of
        # numpy arrays for faster processing and analysis
        # ************************************************************************

        frame_list = []                                                                         # List of numpy arrays
        seconds = 0                                                                             # Number of seconds processed
        frame_count = 0                                                                         # Number of frames processed

        # Iterate through each frame in the video
        while capture.isOpened():

            # Read the frame from the capture
            ret_frame, frame = capture.read()

            # If read successful, then append the retrieved numpy array to a python list
            if ret_frame:
                frame = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_LINEAR)

                # Add the frame to the list
                frame_list.append(frame)

                frame_count += 1
                seconds = frame_count / self.fps
            # Break the loop when frames cannot be taken from original
            else:
                break
            # Break the loop when seconds are longer than defined duration of analysis
            if seconds > self.duration:
                break
        # Clean up memory
        capture.release()

        return np.array(frame_list)

    def compare_renditions_instant(self, frame_pos, frame_list, dimensions, path):
        # ************************************************************************
        # Function to compare pairs of numpy arrays extracting their corresponding metrics.
        # It basically takes the global original frame at frame_pos and its subsequent to
        # compare them against the corresponding ones in frame_list (a rendition).
        # It then extracts the metrics defined in the constructor under the metrics_list.
        # Methods of comparison are implemented in the video_metrics class
        # ************************************************************************

        frame_metrics = {}                                                                      # Dictionary of metrics
        reference_frame = self.original[frame_pos]                                                # Original frame to compare against
        next_reference_frame = self.original[frame_pos + self.skip_frames]                        # Original's subsequent frame
        rendition_frame = frame_list[frame_pos]                                                 # Rendition frame
        next_rendition_frame = frame_list[frame_pos + self.skip_frames]                         # Rendition's subsequent frame

        # Compute the metrics defined in the global metrics_list. Uses the global instance of video_metrics
        # Some metrics use a frame-to-frame comparison, but other require current and forward frames to extract
        # their comparative values.
        rendition_metrics = self.video_metrics.compute_metrics(rendition_frame, next_rendition_frame,
                                                               reference_frame, next_reference_frame)

        # Retrieve rendition dimensions for further evaluation
        rendition_metrics['dimensions'] = dimensions

        # Retrieve rendition ID for further identification
        rendition_metrics['ID'] = path.split('/')[-2]

        # Identify rendition uniquely by its ID and store metric data in frame_metrics dict
        frame_metrics[path] = rendition_metrics

        # Return the metrics, together with the position of the frame
        # frame_pos is needed for the ThreadPoolExecutor optimizations
        return rendition_metrics, frame_pos

    def compute(self, frame_list, path, dimensions):
        # ************************************************************************
        # Function to compare lists of numpy arrays extracting their corresponding metrics.
        # It basically takes the global original list of frames and the input frame_list
        # of numpy arrrays to extract the metrics defined in the constructor.
        # frame_pos establishes the index of the frames to be compared.
        # It is optimized by means of the ThreadPoolExecutor of Python's concurrent package
        # for better parallel performance.
        # ************************************************************************

        rendition_metrics = {}                                                                  # Dictionary of metrics
        frame_pos = 0                                                                           # Position of the frame
        frames_to_process = []                                                                  # List of frames to be processed

        # Iterate frame by frame and fill a list with their values
        # to be passed to the ThreadPoolExecutor. Stop when maximum
        # number of frames has been reached.
        while frame_pos + self.skip_frames < self.duration * self.fps:
            if frame_pos < len(frame_list):
                frames_to_process.append(frame_pos)
            frame_pos += 1

        # Execute computations in parallel using as many processors as possible
        # future_list is a dictionary storing all computed values from each thread
        with ThreadPoolExecutor() as executor:
            # Compare the original asset against its renditions
            future_list = {executor.submit(self.compare_renditions_instant, i, frame_list, dimensions, path): i for i in frames_to_process}

        # Once all frames in frame_list have been iterated, we can retrieve their values
        for future in future_list:
            # Values are retrieved in a dict, as a result of the executor's process
            result_rendition_metrics, frame_pos = future.result()
            # The computed values at a given frame
            rendition_metrics[frame_pos] = result_rendition_metrics

        # Return the metrics for the currently processed rendition
        return rendition_metrics

    def aggregate(self, metrics):
        # ************************************************************************
        # Function to aggregate computed values of metrics and renditions into a
        # pandas DataFrame.
        # ************************************************************************

        metrics_dict = {}                                                                       # Dictionary for containing all metrics
        renditions_dict = {}                                                                    # Dictionary for containing all renditions

        # Aggregate dictionary with all values for all renditions into a Pandas DataFrame
        # All values are stored and obtained in a per-frame basis, then in a per-rendition
        # fashion. They need to be rearranged.

        # First, we combine the frames
        dict_of_df = {k: pd.DataFrame(v) for k, v in metrics.items()}
        metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
        # Pandas concat function creates a level_0 and level_1 extra columns. They need to be renamed
        metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})

        # Then we can combine each rendition
        for rendition in self.renditions_paths:
            # For the current rendition, we need an empty dictionary
            rendition_dict = {}

            # We have a number of different metrics that have been computed.
            # See the list in the video_metrics class
            for metric in self.metrics_list:
                # Obtain a Pandas DataFrame from the original and build the original time series
                original_df = metrics_df[metrics_df['path'] == self.original_path][metric]
                original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)
                # Obtain a Pandas DataFrame from the current rendition and build its time series
                rendition_df = metrics_df[metrics_df['path'] == rendition][metric]
                rendition_df = rendition_df.reset_index(drop=True).transpose().dropna().astype(float)

                # For those metrics that have a temporal character, we need to make a further aggregation
                # We are basically using the Manhattan and euclidean distances, and statistically meaningful
                # values such as mean, max and standard deviation.
                # The whole time series is also provided for later exploration in the analysis part.
                if 'temporal' in metric:
                    x_original = np.array(original_df[rendition_df.index].values)
                    x_rendition = np.array(rendition_df.values)

                    [[manhattan]] = distance.cdist(x_original.reshape(1, -1), x_rendition.reshape(1, -1),
                                                   metric='cityblock')

                    rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original, x_rendition)
                    rendition_dict['{}-manhattan'.format(metric)] = manhattan
                    rendition_dict['{}-mean'.format(metric)] = np.mean(x_rendition)
                    rendition_dict['{}-max'.format(metric)] = np.max(x_rendition)
                    rendition_dict['{}-std'.format(metric)] = np.std(x_rendition)
                    rendition_dict['{}-series'.format(metric)] = x_rendition

                # Other metrics do not need time evaluation
                else:
                    rendition_dict[metric] = rendition_df.mean()

                # Size is an important feature of an asset, as it gives important information
                # regarding the potential compression effect
                rendition_dict['size'] = os.path.getsize(rendition)
                rendition_dict['fps'] = self.fps
                rendition_dict['path'] = rendition

            # Store the rendition values in the dictionary of renditions for the present asset
            renditions_dict[rendition] = rendition_dict

        # Add the current asset values to the global metrics_dict
        metrics_dict[self.original_path] = renditions_dict

        dict_of_df = {k: pd.DataFrame(v) for k, v in metrics_dict.items()}
        metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)

        metrics_df['title'] = metrics_df['level_0']
        attack_series = []
        dimensions_series = []
        for _, row in metrics_df.iterrows():
            attack_series.append(row['level_1'].split('/')[-2])

        metrics_df['attack'] = attack_series

        for _, row in metrics_df.iterrows():
            dimension = int(row['attack'].split('_')[0].replace('p', ''))
            dimensions_series.append(dimension)

        metrics_df['dimension'] = dimensions_series

        metrics_df = metrics_df.drop(['level_0', 'level_1'], axis=1)
        return metrics_df

    def process(self):
        # ************************************************************************
        # Function to aggregate computed values of metrics
        # of iterated renditions into a pandas DataFrame.
        # ************************************************************************
        if self.do_profiling:

            self.capture_to_array = self.cpu_profiler(self.capture_to_array)
            self.compare_renditions_instant = self.cpu_profiler(self.compare_renditions_instant)

        # Iterate through renditions
        for path in self.renditions_paths:
            try:
                capture = cv2.VideoCapture(path)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                dimensions = '{}:{}'.format(int(width), int(height))
                # Turn openCV capture to a list of numpy arrays
                frame_list = self.capture_to_array(capture)
                # Compute the metrics for the rendition
                self.metrics[path] = self.compute(frame_list, path, dimensions)

            except Exception as err:
                print('Unable to compute metrics for {}'.format(path))
                print(err)

        if self.do_profiling:
            self.cpu_profiler.print_stats()

        return self.aggregate(self.metrics)
