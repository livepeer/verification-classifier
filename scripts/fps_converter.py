"""
	This script converts source videos to common framerates.

2020-05-19
"""

import argparse
import math
import subprocess
from os import makedirs
import multiprocessing
from utils import *
import logging
import os
import cv2
import pathlib

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

# Set encoder and decoder based on whether GPU processed is enabled. To use GPU, decoder must be specified explicitly.
decoder = ''
encoder = '-vcodec copy'
if args.gpu:
	decoder = '-hwaccel cuvid -c:v h264_cuvid'
	encoder = '-vcodec h264_nvenc'

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
	files = pathlib.Path(input_path).glob('**/*.*')
	out_dir = pathlib.Path(output_path)
	for f in files:
		dirname = os.path.dirname(str(f)).split(os.path.sep)[-1]
		# skip fps converted renditions
		if 'fps' in dirname:
			continue
		# get source file FPS
		fps = get_fps(str(f))
		if not fps:
			logger.warning(f'Failed to get FPS for file {f}, skipping')
			continue
		fps = round(float(fps))
		# generate output path and metadata for each target fps
		for target_fps in set(rendition_fps).difference({fps}):
			rend_dirname = f'{dirname}_{fps}-{target_fps}fps' + args.suffix
			dst_dir = out_dir / rend_dirname
			dst_filename = dst_dir / f.name
			os.makedirs(dst_dir, exist_ok=True)
			# don't re-encode if rendition already exist
			if not os.path.exists(dst_filename):
				jobs.append((str(f), str(dst_filename), fps, target_fps))
	return jobs


def format_command(input_file, output_file, source_fps, target_fps, encoder, decoder):
	logger.info('processing {}'.format(input_file))
	command = f'ffmpeg -y {decoder} -i {input_file} -vsync 0 -r {target_fps} {encoder} {output_file}'
	return command.split()


def worker(input_file, output_file, source_fps, target_fps):
	ffmpeg_command = ''
	try:
		ffmpeg_command = format_command(input_file, output_file, source_fps, target_fps, encoder, decoder)
		ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
		stdout, stderr = ffmpeg.communicate()
		if ffmpeg.returncode:
			logger.error(f'Error processing file {output_file} with command {ffmpeg_command}: {stderr}')
	except:
		logger.exception(f'Error processing file {output_file} with command {ffmpeg_command}')


if __name__ == "__main__":
	jobs = get_input_output_jobs()
	logger.info(f'Total conversion jobs: {len(jobs)}')
	# nvidia codecs require GPU memory planning and may fail if run in parallel
	with multiprocessing.Pool(cpu_count if not args.gpu else 1) as pool:
		pool.starmap(worker, jobs)
