import cv2
import numpy as np
import time

from video_metrics import video_metrics

class video_asset_processor:
    def __init__(self, source_path, renditions_paths, metrics_list):
        
        # Initialize global variables
        self.source = cv2.VideoCapture(source_path)
        self.chunk_length = 4 * self.source.get(cv2.CAP_PROP_FPS)
        self.asset_length = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.display = False
        self.skip_frames = 1
        self.hash_size = 16
        self.renditions = {}
        self.metrics = {}
        self.metrics_list = metrics_list
        self.video_metrics = video_metrics(self.metrics_list, self.skip_frames, self.hash_size)
        
        # Retrieve original rendition dimensions
        self.height = self.source.get(cv2.CAP_PROP_FRAME_HEIGHT)   
        self.width = self.source.get(cv2.CAP_PROP_FRAME_WIDTH) 
        dimensions = '{}:{}'.format(int(self.width), int(self.height))
        
        # Convert OpenCV video captures of original and renditions to list
        # of numpy arrays for better performance
        self.source = self.capture_to_list(self.source)

        self.renditions['original'] = {'frame_list': self.source,
                                        'dimensions': dimensions,
                                        'ID': source_path.split('/')[-2]}
        # Iterate through renditions
        for path in renditions_paths:
            rendition_ID = path

            capture = cv2.VideoCapture(path)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
            dimensions = '{}:{}'.format(int(width), int(height))

            frame_list = self.capture_to_list(capture)
            self.renditions[rendition_ID] = {'frame_list': frame_list,
                                            'dimensions': dimensions,
                                             'ID': path.split('/')[-2]}
    
    def __del__(self):
        print('Cleaning up')
        # Closes all the frames, in case any were opened by the 'display' flag
        cv2.destroyAllWindows()

    def capture_to_list(self, capture):
        # Initialize 
        start_time = time.time()
        width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))   
        height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_list = []
        frame_count = 0
        frame = np.empty(shape=(width, height), dtype=np.float64)
        # Iterate through each frame in the video
        while capture.isOpened():
            
            # Read the frame from the capture
            ret_frame, frame = capture.read()

            # If read successful, then append the retrieved numpy array to a python list
            if ret_frame:
                # Ensure we are using the luminance space for measuring the reference source
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
                # Add the frame to the list
                frame_list.append(gray)
                frame_count += 1
            # Break the loop when frames cannot be taken from source
            else: 
                elapsed_time = time.time() - start_time 

                break
        # Clean up memory 
        capture.release()

        return frame_list


    def compare_renditions_instant(self, frame_pos):
        # Iterate for each given comparable rendition
        frame_metrics = {}
        count = 0

        reference_frame = self.source[frame_pos]       
        
        
        for rendition_ID, rendition in self.renditions.items():

            count += 1
            rendition_frame_list = rendition['frame_list']
            
            if frame_pos < len(rendition_frame_list):

                start_time = time.time()
                
                rendition_metrics = self.video_metrics.compute_metrics(frame_pos, rendition_frame_list, reference_frame)

                # Collect processing time
                elapsed_time = time.time() - start_time 
                rendition_metrics['time'] = elapsed_time

            # Retrieve rendition dimensions for further evaluation
            rendition_metrics['dimensions'] = rendition['dimensions']
            # Retrieve rendition ID for further identification
            rendition_metrics['ID'] = rendition['ID']
            # Let's identify renditions uniquely by their ID and store their data in frame_metrics dict
            frame_metrics[rendition_ID] = rendition_metrics

            if self.display:
                rendition_frame = rendition_frame_list[frame_pos]
                cv2.imshow(str(count),rendition_frame)

        self.metrics[frame_pos] = frame_metrics

    def process(self):
        # Check if video source opened successfully
        frame_pos = 0

        while frame_pos + self.skip_frames < len(self.source):
            # Compare the original source against its renditions
            self.compare_renditions_instant(frame_pos)
            frame_pos += 1
        return(self.metrics)
        
        