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


class TestAdvAttack:

    @pytest.mark.usefixtures("check_dataset")
    def test_adv_attack(self):
        debug = True
        src = 'testing/tests/data/master2_4s_1080.mp4'
        rend_path = 'testing/tests/data/rend2_4s_1080_adv_attack.mp4'
        verifier = Verifier(10, 'http://storage.googleapis.com/verification-models/verification-metamodel-2020-07-06.tar.xz', False, False, debug)
        verification_result = verifier.verify(src, [{'uri': rend_path}])
        print(verification_result)
        assert verification_result[0]['tamper'] == 1