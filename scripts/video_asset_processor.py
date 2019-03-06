import cv2
import numpy as np
import time
import math
from scipy.spatial import distance
from skimage.measure import compare_ssim

class video_asset_processor:
    
    def __init__(self, source_path, renditions_paths):
        self.source = cv2.VideoCapture(source_path)
        self.display = False
        self.hash_size = 16
        self.chunk_length = 4 * self.source.get(cv2.CAP_PROP_FPS)
        self.asset_length = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.renditions = {}
        self.metrics = {}

        print('Processing asset:', source_path)
        for path in renditions_paths:
            rendition_ID = path
            capture = cv2.VideoCapture(path)
            
            width = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   
            height = capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
            dimensions = '{}:{}'.format(int(width), int(height))

            self.renditions[rendition_ID] = {'capture': capture,
                                            'dimensions': dimensions}
    
    def __del__(self):
        print('Cleaning up')
        self.source.release()
        for key, rendition in self.renditions.items():
            rendition['capture'].release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def rescale_pair(self, img_A, img_B):
        # Limit the scale to the minimum of the dimensions
        width = min(img_A.shape[1], img_B.shape[1])
        height = min(img_A.shape[0], img_B.shape[0])

        resized_A = cv2.resize(img_A, (width, height))
        resized_B = cv2.resize(img_B, (width, height))

        return resized_A, resized_B
        
    def psnr(self, img_A, img_B):
        
        mse = np.mean( (img_A - img_B) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        
    def dhash(self, image):
        # Function to compute the perceptual hash of an image

        # Resize the input image, adding a single column (width) so we
        # can compute the horizontal gradient
        resized = cv2.resize(image, (self.hash_size + 1, self.hash_size))
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]

        # convert the difference image to a hash
        hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        hash_array = [int(x) for x in str(hash)]
        # Return only the first 15 elements of the array
        return hash_array[:15]

    def compare_renditions(self, position_frame, reference_frame, reference_hash):
        # Iterate for each given comparable rendition
        frame_metrics = {}
        count = 0
        for rendition_ID, rendition in self.renditions.items():

            count += 1
            rendition_capture = rendition['capture']
            rendition_capture.set(cv2.CAP_PROP_POS_FRAMES, position_frame)
            ret_rendition, rendition_frame = rendition_capture.read()
            
            # Ensure we are hashing the luminance space
            rendition_frame = cv2.cvtColor(rendition_frame, cv2.COLOR_BGR2GRAY)

            if ret_rendition:
                
                rendition_metrics = {}
                start_time = time.time()

                # Compute the hash of the target frame
                rendition_hash = self.dhash(rendition_frame)

                # Compute different distances with the hash
                rendition_metrics['euclidean'] = distance.euclidean(reference_hash, rendition_hash)
                rendition_metrics['hamming'] = distance.hamming(reference_hash, rendition_hash)
                rendition_metrics['cosine'] = distance.cosine(reference_hash, rendition_hash)

                # Compute SSIM and PSNR
                scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, rendition_frame)
                rendition_metrics['ssim'] = compare_ssim(scaled_reference, scaled_rendition)
                rendition_metrics['psnr'] = self.psnr(scaled_reference, scaled_rendition)

                elapsed_time = time.time() - start_time 
                rendition_metrics['time'] = elapsed_time
                rendition_metrics['dimensions'] = rendition['dimensions']

                frame_metrics[rendition['dimensions']] = rendition_metrics

                if self.display:
                    cv2.imshow(str(count),rendition_frame)
            else:
                print('Rendition not found')
        self.metrics[position_frame] = frame_metrics


    def process(self):
        # Check if video source opened successfully
        frame_count = 0
        if (self.source.isOpened()): 
            position_frame = self.asset_length / 2
            self.source.set(cv2.CAP_PROP_POS_FRAMES, position_frame)
        else:
            print("Error opening video stream or file")

        while(self.source.isOpened() and frame_count < self.chunk_length):
            position_frame = self.source.get(cv2.CAP_PROP_POS_FRAMES) 
            ret, frame = self.source.read()

            # Ensure we are using the luminance space for measuring the reference source
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract the dhash for the reference frame            
            reference_hash = self.dhash(frame)

            if ret:
                # Compare the original source against its renditions
                self.compare_renditions(position_frame, frame, reference_hash)

                if self.display:
                    # Display the frame on the screen
                    cv2.imshow('Frame',frame)
                # Exit the process if q is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print('Exit forced by user. Q pressed')
                    break
            
            # Break the loop when frames cannot be taken from source
            else: 
                print('Finished stream')
                break
            frame_count += 1
        return(self.metrics)
        
        