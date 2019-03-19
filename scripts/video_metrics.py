import numpy as np
import math

class video_metrics:
    def __init__(self, metrics_list, skip_frames,hash_size):
        self.hash_size = hash_size
        self.skip_frames = skip_frames

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
        resized = np.resize(image, (self.hash_size + 1, self.hash_size))
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]
        print(resized)
        # convert the difference image to a hash
        hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        hash_array = [int(x) for x in str(hash)]
        # Return only the first 15 elements of the array
        return hash_array[:15]

    def evaluate_difference_instant(self, frame_list, position_frame):
        # Grab the current frame        
        current_frame = frame_list[position_frame]
        # Grab the frame skip_frames ahead
        next_frame = frame_list[position_frame + self.skip_frames]

        # Compute the number of different pixels
        total_pixels = current_frame.shape[0] * current_frame.shape[1]

        difference_ratio = np.count_nonzero(np.array(next_frame - current_frame)) / total_pixels

        return difference_ratio