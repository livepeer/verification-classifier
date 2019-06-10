import argparse
import subprocess
import sys
from os import makedirs
import multiprocessing
from utils import *

parser = argparse.ArgumentParser(description='Generate renditions')
parser.add_argument('-i', "--input", action='store', help='Folder where the 1080p renditions are', type=str,
                    required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the low bitrate renditions will be', type=str,
                    required=True)
parser.add_argument('-d', "--divisor", action='store', help='Divisor of the current bitrate', type=int, required=True)
parser.add_argument('-m', "--metadata", action='store', help='File where the metadata is', type=str, required=True)
parser.add_argument('-r', "--reprocess", action='store', help='Input file with files to reprocess', type=str,
                    required=False)

args = parser.parse_args()

input_path = args.input
output_path = args.output
bitrate_divisor = args.divisor
metadata_file = args.metadata

if bitrate_divisor == 0:
    print('Divisor can not be 0')
    sys.exit()

cpu_count = multiprocessing.cpu_count()

codec_to_use = 'libx264'

reprocess = False
file_to_reprocess = None

if args.reprocess is not None:
    reprocess = True
    file_to_reprocess = args.reprocess

output_folders = {
    '1080p': '1080p_low_bitrate_{}'.format(bitrate_divisor),
    '720p': '720p_low_bitrate_{}'.format(bitrate_divisor),
    '480p': '480p_low_bitrate_{}'.format(bitrate_divisor),
    '360p': '360p_low_bitrate_{}'.format(bitrate_divisor),
    '240p': '240p_low_bitrate_{}'.format(bitrate_divisor),
    '144p': '144p_low_bitrate_{}'.format(bitrate_divisor),
}

cpu_to_use = 1 if reprocess else int(round(cpu_count / len(output_folders)))

files_and_renditions = get_files_and_renditions(metadata_file)


def crete_folders():
    for key, value in output_folders.items():
        output_folder = output_path + '/' + value
        if not exists(output_folder):
            makedirs(output_folder)


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
        files = get_files_from_file(input_path, file_to_reprocess)
    else:
        files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]
    for file in files:
        bitrates = get_renditions(files_and_renditions[file.split('.mp4')[0]])
        full_input_file = join(input_path, file)
        job_output_folders = {}
        for output_key, output_value in output_folders.items():
            output_folder = join(output_path, output_value)
            full_output_file = join(output_folder, file)
            job_output_folders[output_key] = full_output_file
        ffmpeg_jobs.append((full_input_file, codec_to_use, bitrates, job_output_folders))
    return ffmpeg_jobs


def format_command(full_input_file, codec, bitrates, output_files):
    print('processing {}'.format(full_input_file))
    command = ['ffmpeg', '-y', '-i', '"' + full_input_file + '"',
               '-c:v', codec,
               '-b:v', str(bitrates[1080] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['1080p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:720',
               '-b:v', str(bitrates[720] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['720p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:480',
               '-b:v', str(bitrates[480] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['480p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:360',
               '-b:v', str(bitrates[360] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['360p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:240',
               '-b:v', str(bitrates[240] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['240p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:144',
               '-b:v', str(bitrates[144] / bitrate_divisor) + 'K', '-c:a', 'copy', '"' + output_files['144p'] + '"',
               ]
    return command


def worker(full_input_file, codec, bitrates, output_files):
    ffmpeg_command = ''
    try:
        ffmpeg_command = format_command(full_input_file, codec, bitrates, output_files)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
        if not err:
            print('FFMPEG ERROR')
            print('Out ', out)
            print('Error', err)
    except Exception as e:
        print('Error processing ', full_input_file)
        print('The error was ', e)
        print('Executing ', ffmpeg_command)


if __name__ == "__main__":
    crete_folders()
    jobs = get_input_output_jobs()

    with multiprocessing.Pool(cpu_to_use) as pool:
        pool.starmap(worker, jobs)
