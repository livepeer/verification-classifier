"""
Module for containment of testing procedures
Please try here features and other experiments
that involve video parsing prior to adding them
to the main code body scripts.
"""

import cv2
import numpy as np
import pandas as pd
import math as m
import time
import streamlit as st
from scipy.special import gamma as tgamma


cap = cv2.VideoCapture('../stream/sources/bbb_sunflower_1080p_30fps_normal_t02.mp4')
cap2 = cv2.VideoCapture('../stream/sources/official_test_source_2s_keys_24pfs.mp4')


fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]
time_stamp_counter = 0
frame_counter = 0
max_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
max_samples = int(max_frames * 0.01)
frame_list = np.asarray(sorted(list(np.random.choice(max_frames,
                                          max_samples,
                                          replace=False)))) / fps
start = time.time()
print('Parsing {} out of {} frames'.format(max_samples, max_frames))
while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        frame_counter += 1     
        if frame_counter / fps in frame_list:
            print(frame_counter / fps)    
    else:
        break
print('Time to parse:', time.time() - start)

start = time.time()
for time_stamp in frame_list:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)
    ret, frame = cap.read()
    print(time_stamp)
print('Time to parse:', time.time() -start)

cap.release()
