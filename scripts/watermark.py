import argparse
import csv
import multiprocessing
import subprocess
from os import listdir, makedirs
from os.path import isfile, join, exists
from utils import *

parser = argparse.ArgumentParser(description='Generate renditions with watermarks')
parser.add_argument('-i', "--input", action='store', help='Folder where the 1080p renditions are', type=str,
                    required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the renditions with watermarks will be',
                    type=str, required=True)
parser.add_argument('-m', "--metadata", action='store', help='File where the metadata is', type=str, required=True)
parser.add_argument('-w', "--watermark", action='store', help='Watermark file', type=str, required=True)
parser.add_argument('-s', "--suffix", action='store', help='Watermark folder suffix', type=str, required=True)
parser.add_argument('-x', "--pos_x", action='store', help='Watermark x position (in pixels)', type=str, required=True)
parser.add_argument('-y', "--pos_y", action='store', help='Watermark y position (in pixels)', type=str, required=True)
parser.add_argument('-r', "--reprocess", action='store', help='input file with files to reprocess', type=str,
                    required=False)

args = parser.parse_args()

input_path = args.input
output_path = args.output
metadata_file = args.metadata
watermark_file = args.watermark
pos_x = args.pos_x
pos_y = args.pos_y
reprocess = False
file_to_reprocess = None

if args.reprocess is not None:
    reprocess = True
    file_to_reprocess = args.reprocess

output_folders = {
    '1080': '1080p_watermark-{}'.format(args.suffix),
    '720': '720p_watermark-{}'.format(args.suffix),
    '480': '480p_watermark-{}'.format(args.suffix),
    '360': '360p_watermark-{}'.format(args.suffix),
    '240': '240p_watermark-{}'.format(args.suffix),
    '144': '144p_watermark-{}'.format(args.suffix)
}

cpu_count = multiprocessing.cpu_count()
cpu_to_use = 1 if reprocess else int(round(cpu_count / len(output_folders)))
codec_to_use = 'libx264'

files_and_renditions = {}

with open(metadata_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        files_and_renditions[row[3]] = row[5]


def crete_folders():
    for key, value in output_folders.items():
        folder = output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def format_command(full_input_file, codec, bitrates, output_files):
    print('processing {}'.format(full_input_file))

    command = ['ffmpeg', '-y', '-i', '"' + full_input_file + '"', '-i',
               '"' + watermark_file + '"',
               '-filter_complex',
               '"[0:v]overlay={}:{},split=6[in1][in2][in3][in4][in5][in6];'
               '[in1]scale=-2:1080[out1];[in2]scale=-2:720[out2];[in3]scale=-2:480[out3];[in4]scale=-2:360[out4];'
               '[in5]scale=-2:240[out5];[in6]scale=-2:144[out6]"'.format(pos_x, pos_y),
               '-map', '"[out1]"', '-c:v', codec, '-b:v', str(bitrates[1080]) + 'K', '"' + output_files['1080'] + '"',
               '-map', '"[out2]"', '-c:v', codec, '-b:v', str(bitrates[720]) + 'K', '"' + output_files['720'] + '"',
               '-map', '"[out3]"', '-c:v', codec, '-b:v', str(bitrates[480]) + 'K', '"' + output_files['480'] + '"',
               '-map', '"[out4]"', '-c:v', codec, '-b:v', str(bitrates[360]) + 'K', '"' + output_files['360'] + '"',
               '-map', '"[out5]"', '-c:v', codec, '-b:v', str(bitrates[240]) + 'K', '"' + output_files['240'] + '"',
               '-map', '"[out6]"', '-c:v', codec, '-b:v', str(bitrates[144]) + 'K', '"' + output_files['144'] + '"'
               ]
    return command


def get_files_from_file(input_path, reprocess_file):
    file_list = []
    with open(reprocess_file) as file_reprocess:
        for file_name in file_reprocess:
            full_file = join(input_path, file_name.strip())
            if isfile(full_file):
                file_list.append(file_name.strip())
            else:
                print('File not found {}'.format(full_file))
    print('{} files to reprocess'.format(len(file_list)))
    return file_list


def get_input_output_jobs():
    ffmpeg_jobs = []
    if reprocess:
        job_files = get_files_from_file(input_path, file_to_reprocess)
    else:
        job_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]
    for file in job_files:
        bitrates = get_renditions(files_and_renditions[file.split('.mp4')[0]])
        full_input_file = join(input_path, file)
        job_output_folders = {}
        for output_key, output_value in output_folders.items():
            output_folder = join(output_path, output_value)
            full_output_file = join(output_folder, file)
            job_output_folders[output_key] = full_output_file
        ffmpeg_jobs.append((full_input_file, codec_to_use, bitrates, job_output_folders))
    return ffmpeg_jobs


def worker(full_input_file, codec, bitrates, output_files):
    ffmpeg_command = []
    out = None
    err = None
    try:
        ffmpeg_command = format_command(full_input_file, codec, bitrates, output_files)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
        status = ffmpeg.wait()
        print('FFMPEG status {}'.format(status))
    except Exception as e:
        print('Error processing ', full_input_file)
        print('The error was ', e)
        print('Executing ', ffmpeg_command)
        print('Out ', out)
        print('Error', err)


if __name__ == "__main__":
    crete_folders()
    jobs = get_input_output_jobs()

    with multiprocessing.Pool(cpu_to_use) as pool:
        pool.starmap(worker, jobs)
