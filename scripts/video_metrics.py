import numpy as np
import math
from scipy.spatial import distance
import cv2

class video_metrics:
    def __init__(self, metrics_list, skip_frames, hash_size):
        self.hash_size = hash_size
        self.skip_frames = skip_frames
        self.metrics_list = metrics_list


    def rescale_pair(self, img_A, img_B):
        # Limit the scale to the minimum of the dimensions
        width = min(img_A.shape[0], img_B.shape[0])
        height = min(img_A.shape[1], img_B.shape[1])

        resized_A = cv2.resize(img_A, (height, width))
        resized_B = cv2.resize(img_B, (height, width))

        return resized_A, resized_B

    def mse(self, img_A, img_B):
        return np.mean( (img_A - img_B) ** 2 )

    def psnr(self, img_A, img_B):
        # Function to compute the Peak to Signal Noise Ratio (PSNR)
        # of a pair of images. img_A is considered the original and img_B
        # is treated as the noisy signal
        
        # Compute the Mean Square Error (MSE) between original and copy
        mse = np.mean( (img_A - img_B) ** 2 )

        # Compute PSNR as per definition in Wikipedia: 
        # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        
    def dhash(self, image):
        # Function to compute the perceptual hash of an image

        # Resize the input image, adding a single column (width) so we
        # can compute the horizontal gradient
        resized = np.resize(image, (self.hash_size + 1, self.hash_size))
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]

        # convert the difference image to a hash
        hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        hash_array = [int(x) for x in str(hash)]
        # Return only the first 15 elements of the array
        return hash_array[:15]

    def evaluate_difference_instant(self, frame_list, frame_pos):
        # Function to compute the instantaneous difference between a frame
        # and its subsequent

        # Grab the current frame        
        current_frame = frame_list[frame_pos]
        # Grab the frame skip_frames ahead
        next_frame = frame_list[frame_pos + self.skip_frames]

        # Compute the number of different pixels
        total_pixels = current_frame.shape[0] * current_frame.shape[1]
        
        difference_ratio = np.count_nonzero(np.array(next_frame - current_frame)) / total_pixels

        return difference_ratio
    
    def evaluate_difference_canny_instant(self, reference_frame, next_reference_frame, frame_list, frame_pos):
        # Function to compute the instantaneous difference between a frame
        # and its subsequent, applying a Canny filter
        # Grab the frame skip_frames ahead of the present rendition
        next_frame = frame_list[frame_pos + self.skip_frames]
        scale_width = next_frame.shape[0]
        scale_height = next_frame.shape[1]
        # Rescale input images to fit the rendition's size
        scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, next_frame)   
        scaled_reference_next, scaled_rendition = self.rescale_pair(next_reference_frame, next_frame)  
        
        # Compute the number of different pixels
        total_pixels = scale_width * scale_height

        # Compute the Canny edges for the reference frame, its next frame and the next frame of the rendition
        current_reference_edges = cv2.Canny(scaled_reference,100,200)
        next_reference_edges = cv2.Canny(scaled_reference_next,100,200)
        next_rendition_edges = cv2.Canny(scaled_rendition,100,200)
        
        # Compute the difference between reference frame and its next frame
        reference_difference = np.array(next_reference_edges - current_reference_edges)

        # Compute the difference between reference frame and its corresponding next frame in the rendition
        rendition_difference = np.array(next_rendition_edges - current_reference_edges)

         # Create a kernel for eroding the Canny filtered images
        kernel = np.ones((int(scale_width*0.15), int(scale_height*0.15)), np.uint8)
        # Apply the kernel to dilate and highlight the differences
        reference_dilation = cv2.dilate(reference_difference, kernel, iterations=1)
        rendition_dilation = cv2.dilate(rendition_difference, kernel, iterations=1)

        # Compute the difference ratio between reference and its next
        if np.count_nonzero(reference_dilation) != 0:
            difference_reference_ratio = np.count_nonzero(reference_dilation) / total_pixels
        else:
            difference_reference_ratio = 0.00000001
        # Compute the difference ratio between reference and its next in the rendition
        difference_rendition_ratio = np.count_nonzero(rendition_dilation) / total_pixels

        return difference_rendition_ratio / difference_reference_ratio
    
    def evaluate_psnr_instant(self, reference_frame, frame_list, frame_pos):
        # Function to compute the instantaneous PSNR between a frame
        # and its subsequent within a rendition

        # Grab the frame skip_frames ahead
        next_frame = frame_list[frame_pos + self.skip_frames]

        scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, next_frame)
        difference_psnr = self.psnr(scaled_reference, scaled_rendition)
        return difference_psnr

    def evaluate_mse_instant(self, reference_frame, frame_list, frame_pos):
        # Function to compute the instantaneous difference between a frame
        # and its subsequent

        # Grab the frame skip_frames ahead
        next_frame = frame_list[frame_pos + self.skip_frames]

        scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, next_frame)
        difference_mse = self.mse(scaled_reference, scaled_rendition)
        return difference_mse

    def compute_metrics(self, frame_pos, rendition_frame_list, reference_frame, next_reference_frame):
        rendition_metrics = {}
        for metric in self.metrics_list:
            if metric == 'temporal_difference':
                # Compute the temporal inter frame difference                
                rendition_metrics[metric] = self.evaluate_difference_instant(rendition_frame_list, frame_pos)
            
            if metric == 'temporal_psnr':
                # Compute the temporal inter frame psnr                
                rendition_metrics[metric] = self.evaluate_psnr_instant(reference_frame, rendition_frame_list, frame_pos)
            
            if metric == 'temporal_mse':
                # Compute the temporal inter frame psnr                
                rendition_metrics[metric] = self.evaluate_mse_instant(reference_frame, rendition_frame_list, frame_pos)

            if metric == 'temporal_canny':
                # Compute the temporal inter frame difference of the canny version of the frame
                rendition_metrics[metric] = self.evaluate_difference_canny_instant(reference_frame, next_reference_frame, rendition_frame_list, frame_pos)

            rendition_frame = rendition_frame_list[frame_pos]
            
            # Compute the hash of the target frame
            rendition_hash = self.dhash(rendition_frame)

            # Extract the dhash for the reference frame            
            reference_hash = self.dhash(reference_frame)
            
            # Compute different distances with the hash
            if metric == 'hash_euclidean':
                rendition_metrics['hash_euclidean'] = distance.euclidean(reference_hash, rendition_hash)
            if metric == 'hash_hamming':
                rendition_metrics['hash_hamming'] = distance.hamming(reference_hash, rendition_hash)
            if metric == 'hash_cosine':
                rendition_metrics['hash_cosine'] = distance.cosine(reference_hash, rendition_hash)

            
        return rendition_metrics