import requests
import json
import random
import numpy as np
import pandas as pd
import os
import tqdm
import glob
import verifier
import subprocess
import logging
from video_asset_processor_opencv import VideoAssetProcessorOpenCV
from video_asset_processor import VideoAssetProcessor
import timeit
pd.options.display.width = 0
pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.DEBUG,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()


np.random.seed(123)
random.seed(123)

source_file = '../../data/renditions/1080p/0fIdY5IAnhY_60.mp4'
rendition_file = '../../data/renditions/720p_black_and_white/0fIdY5IAnhY_60.mp4'
n_samples = 30
n_tests = 3
codec = 'libx264'
ver_results = []
transcode_results = []

for i in range(n_tests):
	tc_start = timeit.default_timer()
	args = ['ffmpeg', '-y', '-threads', '1', '-i', source_file,
	 '-c:v', codec, '-vf', 'scale=-2:720',
	 '-b:v', '2000' + 'K', '-c:a', 'copy', '/tmp/out.mp4'
	 ]
	p = subprocess.Popen(args)
	out, err = p.communicate()
	assert not err
	transcode_results.append(timeit.default_timer()-tc_start)

for i in range(n_tests):
	ver_start = timeit.default_timer()
	verifier.retrieve_models('http://storage.googleapis.com/verification-models/verification-metamodel-fps2.tar.xz')
	res = verifier.verify(source_file, [{"uri": rendition_file}], False, n_samples, "/tmp/model", False, False)
	ver_results.append(timeit.default_timer()-ver_start)

ver_time = np.min(ver_results)
transcode_time = np.min(transcode_results)
print(f'Verification time: {ver_time}, SD: {np.std(ver_results)}')
print(f'Transcoding time: {transcode_time}, SD: {np.std(transcode_results)}')
