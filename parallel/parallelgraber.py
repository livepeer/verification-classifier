
import os
import shutil
import timeit
import numpy as np
import cv2
from concurrent.futures.thread import ThreadPoolExecutor

from collections import deque
import multiprocessing

from scripts.asset_processor.video_capture import VideoCapture

class Sample:
    def __init__(self, idx_map, samples, samples_hd, pixels, height, width, fps):
        self.indexes = []
        self.timestamps = []
        self.idx_map = idx_map
        self.samples = samples
        self.samples_hd = samples_hd
        self.pixels = pixels
        self.height = height
        self.width = width
        self.fps = fps

class SampleGraber:
    def __init__(self, id, videopath, use_gpu, do_profiling, debug):
        self.videopath = videopath
        self.use_gpu = use_gpu
        self.do_profiling = do_profiling
        self.debug = debug
        self.debug_frames = False

        self.capture = VideoCapture(videopath)
        self.id = id
        self.fps = self.capture.fps
        self.width = self.capture.width
        self.height = self.capture.height


        if self.debug_frames:
            self.frame_dir_name = type(self).__name__
            shutil.rmtree(self.frame_dir_name, ignore_errors=True)
            os.makedirs(self.frame_dir_name, exist_ok=True)

    @staticmethod
    def _convert_debug_frame(frame):
        return cv2.resize(frame, (1920, 1080), cv2.INTER_CUBIC)

    def grabsamples(self, sampleIdx, timestamps):

        self.indexes = sampleIdx
        self.timestamps = timestamps

        # Create list of random timestamps in video file to calculate metrics at
        # difference between master timestamp and best matching frame timestamp of current video
        timestamp_diffs = [np.inf] * len(self.indexes)
        # currently selected frames
        candidate_frames = [None] * len(self.indexes)
        # maps selected rendition sample to master sample
        debug_index_mapping = {}
        idx_map = []
        frame_list = []
        frame_list_hd = []
        frames_read = 0
        pixels = 0
        height = 0
        width = 0
        timestamps_selected = []
        fps = self.capture.fps
        # Iterate through each frame in the video
        while True:
            # Read the frame from the capture
            frame_data = self.capture.read(grab=True)
            if frame_data is not None:
                frames_read += 1

                # update candidate frames
                ts_diffs = [abs(frame_data.timestamp - mts) for mts in self.timestamps]
                best_match_idx = int(np.argmin(ts_diffs))
                best_match = np.min(ts_diffs)
                # max theoretical timestamp difference between 'matching' frames would be 1/(2*fps) + max(jitter)
                # don't consider frames that are too far, otherwise the algorithm will be linear on memory vs video length
                if best_match < 1 / (2 * self.capture.fps) and timestamp_diffs[best_match_idx] > best_match:
                    timestamp_diffs[best_match_idx] = best_match
                    frame_data = self.capture.retrieve()
                    candidate_frames[best_match_idx] = frame_data
            # Break the loop when frames cannot be taken from original
            else:
                break

        # process picked frames
        for i in range(len(candidate_frames)):
            frame_data = candidate_frames[i]
            ts_diff = timestamp_diffs[i]
            if frame_data is None or ts_diff > 1 / (2 * self.fps):
                # no good matching candidate frame
                continue

            if self.debug_frames:
                cv2.imwrite(
                    f'{self.frame_dir_name}/{i:04}_{"s" if self.self.id  else ""}_{frame_data.index}_{frame_data.timestamp:.4}.png',
                    self._convert_debug_frame(frame_data.frame))


            timestamps_selected.append(frame_data.timestamp)
            idx_map.append(i)
            debug_index_mapping[self.indexes[i]] = frame_data.index
            # Count the number of pixels
            height = frame_data.frame.shape[1]
            width = frame_data.frame.shape[0]
            pixels += height * width

            #if not self.markup_master_frames and self.image_pair_callback is not None:
            #    self.image_pair_callback(self.master_samples_hd[i], frame_data.frame, len(frame_list), ts_diff,
            #                             self.original_path, capture.filename)

            frame_list_hd.append(frame_data.frame)
            # Change color space to have only luminance
            frame = cv2.resize(frame_data.frame, (480, 270), interpolation=cv2.INTER_LINEAR)
            frame_list.append(frame)

        # Clean up memory
        self.capture.release()

        #logger.info(f'Mean master-rendition timestamp diff, sec: {np.mean(
        #    list(filter(lambda x: not np.isinf(x), timestamp_diffs)))} SD: {np.std(
        #    list(filter(lambda x: not np.isinf(x), timestamp_diffs)))}')
        #logger.info(f'Master frame index mapping for {capture.filename}: \n {debug_index_mapping}')

        #return idx_map, np.array(frame_list), np.array(frame_list_hd), pixels, height, width

        sample = Sample(idx_map, np.array(frame_list), np.array(frame_list_hd), pixels, height, width, fps)

        return sample

class ParallelGraber:
    def __init__(self, max_samples, use_gpu, do_profiling, debug):
        """
        Initialize verifier instance
        @param max_samples: Max number of samples to take for a video
        @param model: Either URI of the archive with model files, or local folder path
        @param use_gpu: Use GPU for video decoding and computations
        @param do_profiling: Output execution times to logs
        @param debug: Enable debug image output, greatly reduces performance
        """
        self.max_samples = max_samples
        self.use_gpu = use_gpu
        self.do_profiling = do_profiling
        self.debug = debug

        self.total_frames = 0
        self.sample_indexes = []
        self.sample_timestamps = []
        self.grabers = []
        self.samples = []

    def addgraber(self, videopath):
        id = len(self.grabers)
        graber = SampleGraber(id, videopath, self.use_gpu, self.do_profiling, self.debug)
        self.grabers.append(graber)

        if len(self.grabers) == 1: #master
            self.total_frames = graber.capture.frame_count
            self.sample_indexes = np.sort(np.random.choice(self.total_frames, self.max_samples, False))
            # setting time stamp
            self.sample_timestamps = self.sample_indexes * 1.0 / graber.capture.fps


    def capturesingle(self, id):
        return self.grabers[id].grabsamples(self.sample_indexes, self.sample_timestamps)

    def captureall(self):

        future_result = []
        future_list = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Compare the original asset against its renditions
            for i in range(len(self.grabers)):
                key = f'{i}'
                future = executor.submit(self.capturesingle, i)
                future_list.append((key, future))

        # Once all frames in frame_list have been iterated, we can retrieve their values
        for key, future in future_list:
            # Values are retrieved in a dict, as a result of the executor's process
            frame_pos = future.result()
            future_result.append(frame_pos)

        '''
        for i in range(len(self.grabers)):
            frame_pos = self.grabers[i].grabsamples(self.sample_indexes,self.sample_timestamps)
            future_result.append(frame_pos)
        '''

        # Return the metrics for the currently processed rendition
        return future_result
