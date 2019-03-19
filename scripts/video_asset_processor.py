import cv2
import numpy as np
import time
from scipy.spatial import distance
from skimage.measure import compare_ssim

from video_metrics import video_metrics

class video_asset_processor:
    def __init__(self, source_path, renditions_paths, metrics_list):
        print('Processing asset:', source_path)
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
        print('Building original video capture as list of numpy arrays')
        self.source = self.capture_to_list(self.source)

        self.renditions['original'] = {'frame_list': self.source,
                                        'dimensions': dimensions}
        # Iterate through renditions
        for path in renditions_paths:
            rendition_ID = path

            capture = cv2.VideoCapture(path)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
            dimensions = '{}:{}'.format(int(width), int(height))

            print('Building rendition {} as a list of numpy arrays'.format(rendition_ID))
            frame_list = self.capture_to_list(capture)
            self.renditions[rendition_ID] = {'frame_list': frame_list,
                                            'dimensions': dimensions}
    
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
                print('Finished stream: {} frames processed in {} s'.format(frame_count, elapsed_time))
                break
        # Clean up memory 
        capture.release()

        return frame_list

    def rescale_pair(self, img_A, img_B):
        # Limit the scale to the minimum of the dimensions
        width = min(img_A.shape[1], img_B.shape[1])
        height = min(img_A.shape[0], img_B.shape[0])

        resized_A = cv2.resize(img_A, (width, height))
        resized_B = cv2.resize(img_B, (width, height))

        return resized_A, resized_B

    def compare_renditions_instant(self, position_frame):
        # Iterate for each given comparable rendition
        frame_metrics = {}
        count = 0

        reference_frame = self.source[position_frame]
        # Measure the instantaneous difference between frames
        reference_difference_ratio = self.video_metrics.evaluate_difference_instant(self.source, position_frame)
        # # Extract the dhash for the reference frame            
        # reference_hash = self.dhash(reference_frame)
        
        
        for rendition_ID, rendition in self.renditions.items():

            count += 1
            rendition_frame_list = rendition['frame_list']
            
            if position_frame < len(rendition_frame_list):
                rendition_frame = rendition_frame_list[position_frame]

                rendition_metrics = {}
                start_time = time.time()
                
                # Compute the temporal inter frame difference
                current_frame_ratio = self.video_metrics.evaluate_difference_instant(rendition_frame_list, position_frame)
                if current_frame_ratio != 0:
                    rendition_metrics['temporal_difference'] = current_frame_ratio

                # # Compute the hash of the target frame
                # rendition_hash = self.dhash(rendition_frame)

                # # Compute different distances with the hash
                # rendition_metrics['euclidean'] = distance.euclidean(reference_hash, rendition_hash)
                # rendition_metrics['hamming'] = distance.hamming(reference_hash, rendition_hash)
                # rendition_metrics['cosine'] = distance.cosine(reference_hash, rendition_hash)

                # # Compute SSIM and PSNR
                # scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, rendition_frame)
                # rendition_metrics['ssim'] = compare_ssim(scaled_reference, scaled_rendition)
                # rendition_metrics['psnr'] = self.psnr(scaled_reference, scaled_rendition)

                # Collect processing time
                elapsed_time = time.time() - start_time 
                rendition_metrics['time'] = elapsed_time

            # Retrieve rendition dimensions for further evealuation
            rendition_metrics['dimensions'] = rendition['dimensions']

            # Let's identify renditions uniquely by their ID and store their data in frame_metrics dict
            frame_metrics[rendition_ID] = rendition_metrics

            if self.display:
                cv2.imshow(str(count),rendition_frame)

        self.metrics[position_frame] = frame_metrics

    def process(self):
        # Check if video source opened successfully
        position_frame = 0

        while position_frame + self.skip_frames < len(self.source):
            # Compare the original source against its renditions
            self.compare_renditions_instant(position_frame)
            position_frame += 1
        return(self.metrics)
        
        