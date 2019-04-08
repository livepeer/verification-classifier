import cv2
import numpy as np
import time

from video_metrics import video_metrics
from concurrent.futures.thread import ThreadPoolExecutor


class video_asset_processor:
    def __init__(self, source_path, renditions_paths, metrics_list):
        
        # Initialize global variables
        self.source = cv2.VideoCapture(source_path)
        self.chunk_length = 4 * self.source.get(cv2.CAP_PROP_FPS)
        self.asset_length = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.skip_frames = 1
        self.hash_size = 16
        self.renditions = {}
        self.metrics = {}
        self.metrics_list = metrics_list
        self.video_metrics = video_metrics(self.metrics_list, self.skip_frames, self.hash_size)
        self.renditions_paths = renditions_paths

        # Retrieve original rendition dimensions
        self.height = self.source.get(cv2.CAP_PROP_FRAME_HEIGHT)   
        self.width = self.source.get(cv2.CAP_PROP_FRAME_WIDTH) 
        dimensions = '{}:{}'.format(int(self.width), int(self.height))
        
        # Convert OpenCV video captures of original to list
        # of numpy arrays for better performance of numerical computations
        
        self.source = self.capture_to_list(self.source)
       
        self.renditions['original'] = {'frame_list': self.source,
                                        'dimensions': dimensions,
                                        'ID': source_path.split('/')[-2]}

    def capture_to_list(self, capture):
        print('Converting {} to numpy arrays'.format(capture))
        start_time = time.time()
        frame_list = []
        frame_count = 0

        # Iterate through each frame in the video
        while capture.isOpened():
            
            # Read the frame from the capture
            ret_frame, frame = capture.read()

            # If read successful, then append the retrieved numpy array to a python list
            if ret_frame:
                # Add the frame to the list
                frame_list.append(frame)
                frame_count += 1
            # Break the loop when frames cannot be taken from source
            else:
                break
        # Clean up memory 
        capture.release()

        # Collect processing time
        elapsed_time = time.time() - start_time 
        print('Elapsed time: {}'.format(elapsed_time))
        return np.array(frame_list)

    def compare_renditions_instant(self, frame_pos, frame_list, dimensions, path):
        # Iterate for each given comparable rendition
        frame_metrics = {}

        reference_frame = self.source[frame_pos]       
        next_reference_frame = self.source[frame_pos + self.skip_frames]

        start_time = time.time()
        
        rendition_metrics = self.video_metrics.compute_metrics(frame_pos, frame_list, reference_frame, next_reference_frame)

        # Collect processing time
        elapsed_time = time.time() - start_time 
        rendition_metrics['time'] = elapsed_time

        # Retrieve rendition dimensions for further evaluation
        rendition_metrics['dimensions'] = dimensions
        # Retrieve rendition ID for further identification
        rendition_metrics['ID'] = path.split('/')[-2]

        # Identify rendition uniquely by its ID and store metric data in frame_metrics dict
        frame_metrics[path] = rendition_metrics

        return rendition_metrics, frame_pos

    def compute(self, path):
        rendition_metrics = {}
        capture = cv2.VideoCapture(path)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        dimensions = '{}:{}'.format(int(width), int(height))

        # Turn openCV capture to a list of numpy arrays
        frame_list = self.capture_to_list(capture)

        print('Comparing {}'.format(path))
        start_time = time.time()
        # Iterate frame by frame
        frame_pos = 0
        frames_to_process = []
        while frame_pos + self.skip_frames < len(self.source):
            # Compare the original source against its renditions
            if frame_pos < len(frame_list):
                frames_to_process.append(frame_pos)
            frame_pos += 1

        with ThreadPoolExecutor() as executor:
            future_list = {executor.submit(self.compare_renditions_instant, i, frame_list, dimensions, path): i for i in frames_to_process}

        for future in future_list:
            result_rendition_metrics, frame_pos = future.result()
            rendition_metrics[frame_pos] = result_rendition_metrics

        self.metrics[path] = rendition_metrics
        # Collect processing time
        elapsed_time = time.time() - start_time
        print('{} Comparison elapsed time: {}'.format(path, elapsed_time))

    def process(self):
        # Iterate through renditions
        for path in self.renditions_paths:
            try:
                self.compute(path)
            except:
                print('Unable to compute metrics for {}'.format(path))
        return self.metrics
