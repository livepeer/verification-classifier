"""
Module for management, evaluation and computation of video metrics
"""
import math
import sys
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern as LBP
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.filters import gaussian


class VideoMetrics:
    """
    Class in charge of managing all video metrics for verification on a per-frame basis.
    It wraps up different machine learning and Computer Vision techniques that serve
    to evaluate and extract characteristics of frames of two videos.
    """
    def __init__(self, metrics_list, hash_size, dimension, cpu_profiler, do_profiling):
        self.hash_size = hash_size
        self.metrics_list = metrics_list
        self.dimension = dimension
        self.profiling = do_profiling
        if do_profiling:
            self.cpu_profiler = cpu_profiler

    @staticmethod
    def rescale_pair(reference_frame, rendition_frame):
        """
        Limit the scale to the minimum of the dimensions
        """
        width = min(reference_frame.shape[0], rendition_frame.shape[0])
        height = min(reference_frame.shape[1], rendition_frame.shape[1])

        resized_a = cv2.resize(reference_frame, (height, width))
        resized_b = cv2.resize(rendition_frame, (height, width))

        return resized_a, resized_b

    def mse(self, reference_frame, rendition_frame):
        """
        Function to compute the Mean Square Error (MSE) between two images
        """

        reference_frame, rendition_frame = self.rescale_pair(reference_frame, rendition_frame)
        return np.mean((reference_frame - rendition_frame) ** 2)

    def psnr(self, reference_frame, rendition_frame):
        """
        Function to compute the Peak to Signal Noise Ratio (PSNR)
        of a pair of images. img_A is considered the original and img_B
        is treated as the noisy signal
        """
        reference_frame, rendition_frame = self.rescale_pair(reference_frame, rendition_frame)
        # Compute the Mean Square Error (MSE) between original and copy
        mse = np.mean((reference_frame - rendition_frame) ** 2)

        # Compute PSNR as per definition in Wikipedia:
        # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        if mse == 0:
            return 100
        pixel_max = 255.0
        return 20 * math.log10(pixel_max / math.sqrt(mse))

    def dhash(self, image):
        """
        Function to compute the perceptual hash of an image
        """

        # Resize the input image, adding a single column (width) so we
        # can compute the horizontal gradient
        resized = np.resize(image, (self.hash_size + 1, self.hash_size))
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]

        # convert the difference image to a hash
        image_hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        hash_array = [int(x) for x in str(image_hash)]
        # Return only the first 15 elements of the array
        return hash_array[:15]

    @staticmethod
    def orb(reference_frame, rendition_frame):
        """
        Function to detect and describe keypoints on the first frame,
        It does detect and describe keypoints, then matches them using
        a bruteforce matcher
        ORB is basically a fusion of FAST keypoint detector and
        BRIEF descriptor with many modifications to enhance the performance.
        """
        # Initialize ORB detector

        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        _, descriptor_current = orb.detectAndCompute(reference_frame, None)
        _, descriptor_next = orb.detectAndCompute(rendition_frame, None)

        # create Brute Force Matcher object
        bf_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)


        # Match descriptors.
        matches = bf_matcher.knnMatch(descriptor_current, descriptor_next, k=2)

        if len(np.array(matches).shape) != 2 or np.array(matches).shape[1] != 2:
            return 0

        # Apply ratio test
        good = []
        for match_1, match_2 in matches:
            if 0.50 * match_2.distance < match_1.distance < 0.80 * match_2.distance:
                good.append([match_1])

        # Return the number of matching points between one frame and the next
        return len(good)

    @staticmethod
    def dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """
        max_warping_window = 10000
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - max_warping_window),
                           min(N, i + max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    @staticmethod
    def difference(current_frame, next_frame):
        """
        Function to compute the instantaneous difference between a frame
        and its subsequent
        """

        total_size = current_frame.shape[0] * current_frame.shape[1]
        difference = np.abs(np.float32(next_frame) - np.float32(current_frame))
        difference_ratio = np.mean(difference) / total_size

        return difference_ratio.round(decimals=5)

    @staticmethod
    def entropy(reference_frame, rendition_frame):
        """
        Function that computes the difference in Shannon entropy between
        two images
        """

        entropy_difference = shannon_entropy(reference_frame) - shannon_entropy(rendition_frame)

        return entropy_difference

    @staticmethod
    def lbp(reference_frame, rendition_frame):
        """
        Function that computes the difference in Local Binary patterns between
        two images
        """
        # Settings for LBP
        radius = 3
        n_points = 8 * radius
        method = 'uniform'

        total_pixels = reference_frame.shape[0] * reference_frame.shape[1]

        lbp_reference = LBP(reference_frame, n_points, radius, method)
        lbp_rendition = LBP(rendition_frame, n_points, radius, method)
        lbp_difference = lbp_reference - lbp_rendition

        return np.count_nonzero(lbp_difference) / total_pixels

    @staticmethod
    def spatial_complexity(current_frame):
        """
        # Function to compute the spatial complexity of a video
        """

        sobel_x = cv2.Sobel(current_frame, cv2.CV_64F, 0, 1)
        sobel_y = cv2.Sobel(current_frame, cv2.CV_64F, 1, 0)

        return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

    @staticmethod
    def dct(reference_frame, rendition_frame):
        """
        # Function that computes the Discrete Cosine Transform
        # function included in OpenCV and outputs the
        # Maximum value
        """

        reference_frame_float = np.float32(reference_frame)/255.0  # float conversion/scale
        reference_dct = cv2.dct(reference_frame_float)           # the dct

        rendition_frame_float = np.float32(rendition_frame)/255.0  # float conversion/scale
        rendition_dct = cv2.dct(rendition_frame_float)           # the dct

        _, max_val, _, _ = cv2.minMaxLoc(reference_dct - rendition_dct)

        return max_val

    @staticmethod
    def cross_correlation(reference_frame, rendition_frame):
        """
        # Function that computes the matchTemplate function included in OpenCV and outputs the
        # Maximum value
        """

        # Apply template Matching
        res = cv2.matchTemplate(reference_frame, rendition_frame, cv2.TM_CCORR_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val

    def difference_canny(self, reference_frame, rendition_frame):
        """
        # Function to compute the instantaneous difference between a frame
        # and its subsequent, applying a Canny filter
        """

        # Compute the Canny edges for the reference frame,
        # its next frame and the next frame of the rendition
        lower = 100
        upper = 200

        reference_edges = cv2.Canny(reference_frame, lower, upper)
        rendition_edges = cv2.Canny(rendition_frame, lower, upper)

        return self.mse(reference_edges, rendition_edges)

    @staticmethod
    def ssim(reference_frame, rendition_frame):
        """
        Function to compute the instantaneous SSIM between a frame
        and its correspondent in the rendition
        """

        return structural_similarity(reference_frame,
                                     rendition_frame)

    @staticmethod
    def histogram_distance(reference_frame, rendition_frame, bins=None, eps=1e-10):
        """
        Compute a 3D histogram in the RGB colorspace,
        then normalizes the histogram so that images
        with the same content, but either scaled larger
        or smaller will have (roughly) the same histogram
        """

        if bins is None:
            bins = [8, 8, 8]

        hist_a = cv2.calcHist([reference_frame], [0, 1, 2],
                              None, bins, [0, 256, 0, 256, 0, 256])
        hist_a = cv2.normalize(hist_a, hist_a)
        hist_b = cv2.calcHist([rendition_frame], [0, 1, 2],
                              None, bins, [0, 256, 0, 256, 0, 256])
        hist_b = cv2.normalize(hist_b, hist_b)

        # return out 3D histogram as a flattened array
        hist_a = hist_a.flatten()
        hist_b = hist_b.flatten()

        # Return the chi squared distance of the histograms
        chi_dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hist_a, hist_b)])
        return chi_dist

    @staticmethod
    def gaussian_mse(gauss_reference_frame, gauss_rendition_frame):
        """
        Function that evaluates the mse between a reference
        frame and its rendition.
        Inputs are expected to be the gaussian
        filtered version of the frames.
        """

        mse = mean_squared_error(gauss_reference_frame, gauss_rendition_frame)
        return mse

    @staticmethod
    def gaussian_difference(gauss_reference_frame, gauss_rendition_frame):
        """
        Function that evaluates the total sum of the difference between a reference
        frame and its rendition.
        Inputs are expected to be the gaussian filtered version of the frames.
        """

        difference = np.abs(np.float32(gauss_reference_frame - gauss_rendition_frame))

        return np.sum(difference)

    @staticmethod
    def gaussian_difference_threshold(gauss_reference_frame,
                                      gauss_rendition_frame,
                                      rendition_frame,
                                      next_reference_frame):
        """
        Function that evaluates the total sum of the number of pixels above a
        threshold that is defined by the difference between a reference
        frame and its rendition.
        The threshold is defined as the standard deviation of the difference between
        frames prior to the gaussian filter.
        """
        # Normalize the input frames by dividing between 255 and make the subtraction
        # between the next source frame and the current in the rendition
        temporal_difference = (next_reference_frame / 255) - (rendition_frame / 255)
        # Convert the difference to its absolute value
        temporal_difference = np.abs(np.float32(temporal_difference))

        # Compute the difference between current rendition frame and the reference
        difference = np.abs(np.float32(gauss_reference_frame - gauss_rendition_frame))

        _, threshold = cv2.threshold(difference, temporal_difference.std(), 1, cv2.THRESH_BINARY)

        sum_th = np.sum(threshold)

        return sum_th

    @staticmethod
    def texture_instant(reference_frame, rendition_frame):
        """
        Haralick features date back to as far as 1970s and were one 
        of the first used to classify aerial imagery collected from satellites.

        The idea behind haralick feature extraction is to:

        1.- Compute co-occurence matrix from the image 
        (generated by counting the number of times a pixel with 
        value i is adjacent to a pixel with value j)
        2.- Compute statistics of the matrix like contrast, correlation, variation etc.
        Credit on the above must be given to:
        http://kampta.github.io/Performance-Shootout-mahotas-vs-skimage-vs-opencv-part2/
        """
        # 1.- Compute co-occurence matrix from the reference image
        reference_frame = greycomatrix(reference_frame, 
                                       range(4),
                                       np.pi/4*np.arange(4),
                                       levels=256,
                                       symmetric=True,
                                       normed=True)
        # 2.- Compute co-occurence matrix from the rendition image
        rendition_frame = greycomatrix(rendition_frame, 
                                       range(4),
                                       np.pi/4*np.arange(4),
                                       levels=256,
                                       symmetric=True,
                                       normed=True)                                                       
        # 3.- Compute statistics of the matrix like contrast, correlation, variation
        reference_texture = greycoprops(reference_frame)
        rendition_texture = greycoprops(rendition_frame)

        return mean_squared_error(reference_texture, rendition_texture)

    @staticmethod
    def image_match_instant(pixelsA, pixelsB, v):  
        """
        Original go implementation:
        https://github.com/mkrufky/coersion/
        
        func imageMatch(wg *sync.WaitGroup, pixelsA, pixelsB [][]Pixel, v uint64, pass, fail *int) {
            defer wg.Done()
            *pass = 0
            *fail = 0
            for x, a := range pixelsA {
                for y, b := range a {
                    if math.Abs(float64(b.R-pixelsB[x][y].R)) < float64(v) &&
                        math.Abs(float64(b.G-pixelsB[x][y].G)) < float64(v) &&
                        math.Abs(float64(b.B-pixelsB[x][y].B)) < float64(v) &&
                        math.Abs(float64(b.A-pixelsB[x][y].A)) < float64(v) {
                        *pass++
                    } else {
                        *fail++
                    }
                }
            }
        }
        """


        pass_count = np.sum(np.abs(np.float64(pixelsA - pixelsB)) < v)
        
        return pass_count / np.size(pixelsA)

    @staticmethod
    def brisque_features(reference_frame):
        """
        Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE)
        is a natural scene statistic (NSS)-based distortion-generic
        blind/no-reference (NR) image quality assessment (IQA) model
        which operates in the spatial domain.
        It does not compute distortion specific features such as ringing,
        blur or blocking, but instead uses scene statistics of locally
        normalized luminance coefficients to quantify possible losses
        of ‘naturalness’ in the image due to the presence of distortions,
        thereby leading to a holistic measure of quality.
        """
        features = np.empty([36,])
        
        features = cv2.quality.QualityBRISQUE_computeFeatures(reference_frame, features)

        return features

    def compute_metrics(self,
                        rendition_frame,
                        next_rendition_frame,
                        reference_frame,
                        next_reference_frame,
                        rendition_frame_HD=None,
                        reference_frame_HD=None):

        if self.profiling:
            self.cross_correlation = self.cpu_profiler(self.cross_correlation)
            self.dct = self.cpu_profiler(self.dct)
            self.entropy = self.cpu_profiler(self.entropy)
            self.lbp = self.cpu_profiler(self.lbp)
            self.difference_canny = self.cpu_profiler(self.difference_canny)
            self.difference = self.cpu_profiler(self.difference)
            self.spatial_complexity = self.cpu_profiler(self.spatial_complexity)
            self.gaussian_mse = self.cpu_profiler(self.gaussian_mse)
            self.gaussian_difference = self.cpu_profiler(self.gaussian_difference)
            self.gaussian_difference_threshold = self.cpu_profiler(self.gaussian_difference_threshold)
            self.mse = self.cpu_profiler(self.mse)
            self.psnr = self.cpu_profiler(self.psnr)
            self.ssim = self.cpu_profiler(self.ssim)
            self.orb = self.cpu_profiler(self.orb)
            self.rescale_pair = self.cpu_profiler(self.rescale_pair)
            self.texture_instant = self.cpu_profiler(self.texture_instant)
            self.image_match_instant = self.cpu_profiler(self.image_match_instant)
            self.brisque_features = self.cpu_profiler(self.brisque_features)

        rendition_metrics = {}
        # Some metrics only need the luminance channel
        reference_frame_gray = reference_frame
        rendition_frame_gray = rendition_frame
        next_reference_frame_gray = next_reference_frame
        next_rendition_frame_gray = next_rendition_frame

        sigma = 4
        gauss_reference_frame = gaussian(reference_frame_gray, sigma=sigma)
        gauss_rendition_frame = gaussian(rendition_frame_gray, sigma=sigma)

        for metric in self.metrics_list:
            if metric == 'temporal_brisque':
                rendition_metrics[metric] = self.brisque_features(rendition_frame)

            if metric == 'temporal_histogram_distance':
                rendition_metrics[metric] = self.histogram_distance(reference_frame,
                                                                    rendition_frame)

            if metric == 'temporal_difference':
                rendition_metrics[metric] = self.difference(rendition_frame_gray,
                                                            next_rendition_frame_gray)

            if metric == 'temporal_orb':
                rendition_metrics[metric] = self.orb(reference_frame_gray,
                                                     rendition_frame_gray)

            if metric == 'temporal_psnr':
                rendition_metrics[metric] = self.psnr(reference_frame_HD,
                                                      rendition_frame_HD)

            if metric == 'temporal_ssim':
                rendition_metrics[metric] = self.ssim(reference_frame_HD,
                                                      rendition_frame_HD)

            if metric == 'temporal_mse':
                rendition_metrics[metric] = self.mse(reference_frame_gray,
                                                     rendition_frame_gray)

            if metric == 'temporal_canny':
                rendition_metrics[metric] = self.difference_canny(reference_frame_gray,
                                                                  rendition_frame_gray)

            if metric == 'temporal_cross_correlation':
                rendition_metrics[metric] = self.cross_correlation(reference_frame_gray,
                                                                   rendition_frame_gray)

            if metric == 'temporal_dct':
                rendition_metrics[metric] = self.dct(reference_frame_gray,
                                                     rendition_frame_gray)

            if metric == 'temporal_gaussian_mse':
                rendition_metrics[metric] = self.gaussian_mse(gauss_reference_frame,
                                                              gauss_rendition_frame)

            if metric == 'temporal_gaussian_difference':
                rendition_metrics[metric] = self.gaussian_difference(gauss_reference_frame,
                                                                     gauss_rendition_frame)

            if metric == 'temporal_threshold_gaussian_difference':
                rendition_metrics[metric] = self.gaussian_difference_threshold(gauss_reference_frame,
                                                                               gauss_rendition_frame,
                                                                               rendition_frame_gray,
                                                                               next_reference_frame_gray)
            if metric == 'temporal_spatial_complexity':
                rendition_metrics[metric] = self.spatial_complexity(reference_frame_gray)

            if metric == 'temporal_texture':
                rendition_metrics[metric] = self.texture_instant(reference_frame_gray, rendition_frame_gray)

            if metric == 'temporal_match':
                match_threshold = 10
                rendition_metrics[metric] = self.image_match_instant(reference_frame_gray, rendition_frame_gray, match_threshold)

            if metric == 'temporal_entropy':
                rendition_metrics[metric] = self.entropy(reference_frame_gray,
                                                         rendition_frame_gray)

            if metric == 'temporal_lbp':
                rendition_metrics[metric] = self.lbp(reference_frame_gray,
                                                     rendition_frame_gray)

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
