import numpy as np
import math
from scipy.spatial import distance
from skimage.measure import compare_ssim

class video_metrics:
    def __init__(self, metrics_list, skip_frames, hash_size):
        self.hash_size = hash_size
        self.skip_frames = skip_frames
        self.metrics_list = metrics_list


    def rescale_pair(self, img_A, img_B):
        # Limit the scale to the minimum of the dimensions
        width = min(img_A.shape[1], img_B.shape[1])
        height = min(img_A.shape[0], img_B.shape[0])

        resized_A = np.resize(img_A, (width, height))
        resized_B = np.resize(img_B, (width, height))

        return resized_A, resized_B

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

    def evaluate_difference_instant(self, frame_list, position_frame):
        # Function to compute the instantaneous difference between a frame
        # and its subsequent

        # Grab the current frame        
        current_frame = frame_list[position_frame]
        # Grab the frame skip_frames ahead
        next_frame = frame_list[position_frame + self.skip_frames]

        # Compute the number of different pixels
        total_pixels = current_frame.shape[0] * current_frame.shape[1]
        
        difference_ratio = np.count_nonzero(np.array(next_frame - current_frame)) / total_pixels

        return difference_ratio
    
    def compute_metrics(self, position_frame, rendition_frame_list, reference_frame):
        rendition_metrics = {}
        for metric in self.metrics_list:
            if metric == 'temporal_difference':
                # Compute the temporal inter frame difference                
                rendition_metrics['temporal_difference'] = self.evaluate_difference_instant(rendition_frame_list, position_frame)

            rendition_frame = rendition_frame_list[position_frame]
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

            # Compute SSIM and PSNR
            scaled_reference, scaled_rendition = self.rescale_pair(reference_frame, rendition_frame)
            if metric == 'ssim':
                rendition_metrics['ssim'] = compare_ssim(scaled_reference, scaled_rendition)
            if metric == 'psnr':
                rendition_metrics['psnr'] = self.psnr(scaled_reference, scaled_rendition)
        return rendition_metrics