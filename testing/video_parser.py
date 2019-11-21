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


cap = cv2.VideoCapture('../stream/sources/vimeo/104681073-no-audio.mp4')
cap2 = cv2.VideoCapture('../stream/sources/vimeo/104681073.mp4')

start_time = time.time()
i = 0
while cap.isOpened():
    ret1, frame1 = cap.read()
    # ret2, frame2 = cap2.read()
    if ret1:# and ret2:
        i += 1
        # res = cv2.quality.QualityMSE_compute(frame1, frame2)
        if i == 5:
            features = np.empty([36,])
            # cv2.imshow('Frame', frame1)
            # cv2.waitKey(0)
            features = cv2.quality.QualityBRISQUE_computeFeatures(frame1, features) # specify model_path and range_path
            i = 0        
        # print(i, score)
    else:
        break
print(features)
print('Elapsed time:', time.time() - start_time)
