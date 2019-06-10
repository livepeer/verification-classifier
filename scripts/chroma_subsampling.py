import argparse
import subprocess
from os import makedirs
import multiprocessing
from utils import *

parser = argparse.ArgumentParser(description='Generate renditions')
parser.add_argument('-i', "--input", action='store', help='Folder where the 1080p renditions are', type=str,
                    required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the vignetted renditions will be',
                    type=str, required=True)
parser.add_argument('-s', "--subsampling", action='store', help='Desired Chroma Subsampling', type=str, required=True)
parser.add_argument('-m', "--metadata", action='store', help='File where the metadata is', type=str, required=True)

args = parser.parse_args()

input_path = args.input
output_path = args.output
subsampling = args.subsampling
metadata_file = args.metadata


cpu_count = multiprocessing.cpu_count()

codec_to_use = 'libx264'

output_folders = {
    '720p': '720p_chroma_subsampling',
    '480p': '480p_chroma_subsampling',
    '360p': '360p_chroma_subsampling',
    '240p': '240p_chroma_subsampling',
    '144p': '144p_chroma_subsampling',
}

cpu_to_use = int(round(cpu_count / len(output_folders)))

files_and_renditions = get_files_and_renditions(metadata_file)

subsampling_suffix = '_' + subsampling
for folder, new_folder in output_folders.items():
    output_folders[folder] = output_folders[folder] + subsampling_suffix
ffmpeg_filter = 'format=' + subsampling


def crete_folders():
    for key, value in output_folders.items():
        output_folder = output_path + '/' + value
        if not exists(output_folder):
            makedirs(output_folder)


def get_input_output_jobs():
    ffmpeg_jobs = []
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
               '-c:v', codec, '-vf', 'scale=-2:720,{}'.format(ffmpeg_filter),
               '-b:v', str(bitrates[720]) + 'K', '-c:a', 'copy', '"' + output_files['720p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:480,{}'.format(ffmpeg_filter),
               '-b:v', str(bitrates[480]) + 'K', '-c:a', 'copy', '"' + output_files['480p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:360,{}'.format(ffmpeg_filter),
               '-b:v', str(bitrates[360]) + 'K', '-c:a', 'copy', '"' + output_files['360p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:240,{}'.format(ffmpeg_filter),
               '-b:v', str(bitrates[240]) + 'K', '-c:a', 'copy', '"' + output_files['240p'] + '"',
               '-c:v', codec, '-vf', 'scale=-2:144,{}'.format(ffmpeg_filter),
               '-b:v', str(bitrates[144]) + 'K', '-c:a', 'copy', '"' + output_files['144p'] + '"',
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
