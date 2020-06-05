"""
	This script converts source videos to common framerates.

2020-05-19
"""
import random
import argparse
import math
import re
import subprocess
from os import makedirs
import multiprocessing

from tqdm import tqdm
from utils import *
import logging
import itertools
import os
import cv2
import pathlib
import numpy as np
from imblearn.over_sampling import RandomOverSampler

# init logging
logging.basicConfig(level=logging.DEBUG,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Performs common framerate conversions on videos')
parser.add_argument('-i', "--input", action='store', help='Folder containing folders with rendition video files.', type=str,
					required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the renditions will be', type=str,
					required=True)
parser.add_argument('-g', "--gpu", action='store_true', help='Use hardware codecs. Supported only for H264 input and output. Make sure Ffmpeg is compiled to support Nvidia hardware codecs.', required=False, default=False)

parser.add_argument('-s', "--suffix", action='store', help='Add suffix to output folder name', required=False, default='')

args = parser.parse_args()

cpu_count = multiprocessing.cpu_count()

input_path = args.input
output_path = args.output

# Number of rendition kinds in input folder. Used to generate same number of FPS-adjusted renditions as other renditions.
rendition_type_number = 12

# target FPS
rendition_fps = [60, 30, 24]


def get_fps(filename):
	fps = None
	try:
		cap = cv2.VideoCapture(filename)
		if cap.isOpened():
			fps = cap.get(cv2.CAP_PROP_FPS)
	finally:
		try:
			cap.release()
		except:
			pass
	return fps


def get_input_output_jobs():
	jobs = []
	# get all video files
	files = list(pathlib.Path(input_path).glob('**/*.*'))
	out_dir = pathlib.Path(output_path)

	def non_fps_file(f):
		dirname = os.path.dirname(str(f)).split(os.path.sep)[-1]
		return 'fps' not in dirname
	fps_rend_files = list(itertools.filterfalse(non_fps_file, files))
	files = list(filter(non_fps_file, files))
	master_files = []
	for f in tqdm(files, 'Reading metadata...'):
		# get source file FPS
		dirname = os.path.dirname(str(f)).split(os.path.sep)[-1]
		fps = get_fps(str(f))
		if not fps:
			logger.warning(f'Failed to get FPS for file {f}, skipping')
			continue
		fps = round(float(fps))
		if fps not in [24, 25, 30, 50, 60]:
			logger.warning(f'{f} FPS is {fps}, skipping')
			continue
		master_files.append(f)
		# generate output path and metadata for each target fps
		for target_fps in set(rendition_fps).difference({fps}):
			gpu = False
			suffix = ''
			if random.random() > 0.5:
				gpu = True
				suffix = '_gpu'
			rend_dirname = f'{dirname}_{fps}-{target_fps}fps' + suffix
			dst_dir = out_dir / rend_dirname
			dst_filename = dst_dir / f.name
			os.makedirs(dst_dir, exist_ok=True)
			# don't re-encode if rendition already exist
			if not os.path.exists(dst_filename):
				jobs.append((str(f), str(dst_filename), fps, target_fps, gpu))
	jobs_np = np.array(jobs)
	# resample by fps to make random choice probability equal
	jobs_bal, _ = RandomOverSampler().fit_resample(jobs_np, jobs_np[:, 2])
	# select 1/number_of_renditions of all videos
	jobs_subsample = jobs_bal[np.random.choice(jobs_bal.shape[0], (len(files) // rendition_type_number) - len(fps_rend_files), False)]
	jobs = np.unique(jobs_subsample, axis=0).tolist()
	return jobs


def format_command(input_file, output_file, source_fps, target_fps, encoder, decoder, bitrate):
	logger.info('processing {}'.format(input_file))
	command = f'ffmpeg -y {decoder} -i {input_file} -filter:v fps=fps={target_fps} -b:v {bitrate}K {encoder} {output_file}'
	return command.split()


def worker(input_file, output_file, source_fps, target_fps, gpu):
	ffmpeg_command = ''
	try:
		# detect bitrate
		ffprobe = subprocess.Popen(f'ffprobe -i {input_file}'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		metadata_str = ffprobe.stderr.read().decode('ascii')
		m = re.search('.+bitrate: (?P<bitrate>\d*)', metadata_str)
		bitrate = int(m.group('bitrate'))
		logger.info(f'Video bitrate is: {bitrate}')

		decoder = ''
		encoder = '-vcodec libx264'
		if gpu:
			decoder = '-hwaccel cuvid -c:v h264_cuvid'
			encoder = '-vcodec h264_nvenc'
		ffmpeg_command = format_command(input_file, output_file, source_fps, target_fps, encoder, decoder, bitrate)
		ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
		stdout, stderr = ffmpeg.communicate()
		if ffmpeg.returncode:
			logger.error(f'Error processing file {output_file} with command {ffmpeg_command}: {stderr}')
		ffmpeg.terminate()
	except:
		logger.exception(f'Error processing file {output_file} with command {ffmpeg_command}')


if __name__ == "__main__":
	jobs = get_input_output_jobs()
	logger.info(f'Total conversion jobs: {len(jobs)}')
	# NVENC has session number limitations based on GPU
	with multiprocessing.Pool(4) as pool:
		pool.starmap(worker, jobs, chunksize=100)
