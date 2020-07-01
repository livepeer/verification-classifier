import requests
import json
import random
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import glob
import logging
from scripts.asset_processor import VideoAssetProcessor, VideoCapture
import timeit
import pytest

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

class PairWriter():
    def __init__(self, out_path, is_tamper, img_size):
        self.out_path = out_path
        self.is_tamper = is_tamper
        self.img_size = img_size

    def pair_callback(self, master, rend, idx, ts_diff, master_path, rend_path):
        name_base = f'{master_path.split(os.sep)[-1].split(".")[0]}__{master.shape[1]}__{rend_path.split(os.sep)[-2]}__{ts_diff:.2f}__{idx}'
        full_name = os.path.join(self.out_path, 'tamper' if self.is_tamper else 'correct', name_base)
        master = cv2.resize(master, self.img_size)
        rend = cv2.resize(rend, self.img_size)
        cv2.imwrite(full_name+'__m.png', master)
        cv2.imwrite(full_name+'__r.png', rend)


def main():
    # switch work dir to project root in case tests are run directly
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../..')
    source_dir = '../data/renditions/1080p/'
    out_dir = '../data/cnn/'
    img_size = (160, 160)
    rendition_dirs = [
        ('../data/renditions/720p_60-24fps/', False),
        ('../data/renditions/720p_watermark/', True),
        ('../data/renditions/720p_vignette/', True),
        ('../data/renditions/720p_rotate_90_clockwise/', True),
        ('../data/renditions/720p_black_and_white/', True),

        ('../data/renditions/480p_60-24fps/', False),
        ('../data/renditions/480p_vignette/', True),
        ('../data/renditions/480p_watermark/', True),
        ('../data/renditions/480p_rotate_90_clockwise/', True),
        ('../data/renditions/480p_black_and_white/', True),

        ('../data/renditions/720p/', False),
        ('../data/renditions/480p/', False),
    ]
    files = None
    debug = False
    n_samples = 10
    gpu = False
    src_videos = sorted(glob.glob(source_dir + '/*'))
    results = []

    for src in tqdm.tqdm(src_videos):
        filename = src.split(os.path.sep)[-1]
        if files is not None and not filename in files:
            continue
        i = 0
        for rendition_dir, tamper in rendition_dirs:
            rendition_name = rendition_dir.strip(os.path.sep).split(os.path.sep)[-1]
            rend_path = os.path.join(rendition_dir, filename)
            if not os.path.exists(rend_path):
                continue
            np.random.seed(123)
            random.seed(123)
            pv = PairWriter(out_dir, tamper, img_size)
            vap = VideoAssetProcessor({'path': src}, [{'path': rend_path}], [], False, n_samples, [], debug, False, image_pair_callback=pv.pair_callback)
            vap.process()
