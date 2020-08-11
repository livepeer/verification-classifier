import requests
import json
import random
import numpy as np
import pandas as pd
import os
import tqdm
import glob
from verifier import Verifier
import logging
from scripts.asset_processor import VideoAssetProcessor, VideoCapture
import timeit
import pytest

pd.options.display.width = 0
pd.set_option('display.max_columns', None)


class TestVarFps:

    def test_opencv_pts_validity(self):
        filename = 'testing/tests/data/0fIdY5IAnhY_60.mp4'
        # read timestamps with opencv
        cap = VideoCapture(filename, video_reader='opencv')
        while True:
            f = cap.read()
            if f is None:
                break
        cap.release()
        opencv_ts = cap.timestamps
        # read timestamps with ffmpeg
        cap = VideoCapture(filename, video_reader='ffmpeg_bgr')
        while True:
            f = cap.read()
            if f is None:
                break
        cap.release()
        ffmpeg_ts = cap.timestamps
        assert all(np.isclose(ffmpeg_ts, opencv_ts, atol=0.001))

    @pytest.mark.usefixtures("check_dataset")
    def test_classification(self):
        source_dir = '../data/renditions/1080p/'
        rendition_dirs = [
            ('../data/renditions/720p_watermark/', True),
            ('../data/renditions/720p_60-24fps/', False),
        ]
        files = None
        debug = False
        n_samples = 10
        gpu = False
        src_videos = sorted(glob.glob(source_dir + '/*'))
        results = []
        verifier = Verifier(n_samples, 'http://storage.googleapis.com/verification-models/verification-metamodel-2020-07-06.tar.xz', gpu, False, debug)
        for src in tqdm.tqdm(src_videos):
            filename = src.split(os.path.sep)[-1]
            if files is not None and not filename in files:
                continue
            i = 0
            for rendition_dir, tamper in rendition_dirs:
                rendition_name = rendition_dir.strip(os.path.sep).split(os.path.sep)[-1]
                rend_path = rendition_dir + os.path.sep + filename
                if not os.path.exists(rend_path):
                    continue
                np.random.seed(123)
                random.seed(123)
                verification_result = verifier.verify(src, [{'uri': rend_path}])
                score_meta = float(verification_result[0]["tamper"])
                score_ul = float(verification_result[0]["tamper_ul"])
                score_sl = float(verification_result[0]["tamper_sl"])
                res = {'score': score_meta, 'score_ul': score_ul, 'score_sl': score_sl}
                res['master_filename'] = filename
                res['rendition_type'] = rendition_name
                res['is_tamper'] = tamper
                results.append(res)
        df_res: pd.DataFrame = pd.DataFrame(results)
        df_res.set_index(['master_filename', 'rendition_type'], inplace=True)
        df_res.sort_index(inplace=True)
        df_res['prediction'] = df_res['score'] > 0
        print(df_res)
        # assert accuracy
        assert np.sum(df_res.prediction == df_res.is_tamper)/len(df_res) >= 0.8