import requests
import json
import random
import numpy as np
import pandas as pd
import os
import tqdm
import glob
from parallel.parallelverifier import ParallelVerifier
import logging
from scripts.asset_processor import VideoAssetProcessor, VideoCapture
import timeit
import pytest
import cProfile

pd.options.display.width = 0
pd.set_option('display.max_columns', None)

class TestParallelDecode:
    
    def test_parallel_decodetamper(self):
        debug = False
        src = 'testing/tests/data/master2_4s_1080.mp4'
        rend_path = 'testing/tests/data/rend2_4s_1080_adv_attack.mp4'
        verifier = ParallelVerifier(10, 'http://storage.googleapis.com/verification-models/verification-metamodel-2020-07-06.tar.xz', False, False, debug)
        verification_result = verifier.verify(src, [{'uri': rend_path}])
        print(verification_result)
        assert verification_result[0]['tamper'] == 1

 