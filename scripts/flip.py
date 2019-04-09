import argparse
import multiprocessing
import subprocess
from os import makedirs
from utils import *

parser = argparse.ArgumentParser(description='Generate fliped renditions')
parser.add_argument('-i', '--input', action='store', help='Folder where the 1080p renditions are', type=str,
                    required=True)
parser.add_argument('-o', '--output', action='store', help='Folder where the fliped renditions will be',
                    type=str, required=True)
parser.add_argument('-m', "--metadata", action='store', help='File where the metadata is', type=str, required=True)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-vf', '--vflip', action='store_true', help='Flip video vertically')
group.add_argument('-hf', '--hflip', action='store_true', help='Flip video horizontally')
group.add_argument('-cf', '--clockflip', action='store_true', help='Rotate 90 degrees clockwise')
group.add_argument('-ccf', '--counterclockflip', action='store_true', help='Rotate 90 degrees counterclockwise')

args = parser.parse_args()

arg_input_path = args.input
arg_output_path = args.output
vertical_flip = args.vflip
horizontal_flip = args.hflip
clockwise_flip = args.clockflip
counterclockwise_flip = args.counterclockflip
metadata_file = args.metadata

output_folders = {
    'vertical_flip': {
        '1080': '1080p_flip_vertical',
        '720': '720p_flip_vertical',
        '480': '480p_flip_vertical',
        '360': '360p_flip_vertical',
        '240': '240p_flip_vertical',
        '144': '144p_flip_vertical',
    },
    'horizontal_flip': {
        '1080': '1080p_flip_horizontal',
        '720': '720p_flip_horizontal',
        '480': '480p_flip_horizontal',
        '360': '360p_flip_horizontal',
        '240': '240p_flip_horizontal',
        '144': '144p_flip_horizontal',
    },
    'clockwise': {
        '1080': '1080p_rotate_90_clockwise',
        '720': '720p_rotate_90_clockwise',
        '480': '480p_rotate_90_clockwise',
        '360': '360p_rotate_90_clockwise',
        '240': '240p_rotate_90_clockwise',
        '144': '144p_rotate_90_clockwise',
    },
    'counterclockwise': {
        '1080': '1080p_rotate_90_counterclockwise',
        '720': '720p_rotate_90_counterclockwise',
        '480': '480p_rotate_90_counterclockwise',
        '360': '360p_rotate_90_counterclockwise',
        '240': '240p_rotate_90_counterclockwise',
        '144': '144p_rotate_90_counterclockwise',
    }
}

command_modifier = {
    'vertical_flip': 'vflip',
    'horizontal_flip': 'hflip',
    'clockwise': 'transpose=1',
    'counterclockwise': 'transpose=2'
}


def selected_bool_to_str():
    if vertical_flip:
        return 'vertical_flip'
    if horizontal_flip:
        return 'horizontal_flip'
    if clockwise_flip:
        return 'clockwise'
    if counterclockwise_flip:
        return 'counterclockwise'


files_and_renditions = get_files_and_reinditions(metadata_file)

selected_option = selected_bool_to_str()

cpu_count = multiprocessing.cpu_count()
cpu_to_use = int(round(cpu_count / len(output_folders[selected_option])))
codec_to_use = 'libx264'


def crete_folders():
    for key, value in output_folders[selected_option].items():
        folder = arg_output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def format_command(full_input_file, codec, bitrates, output_files):
    print('processing {}'.format(full_input_file))
    modifier = command_modifier[selected_option]

    command = ['ffmpeg', '-y',
               '-i', '"' + full_input_file + '"',
               '-vf', 'scale=-2:1080,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[1080]) + 'K',
               '"' + output_files['1080'] + '"',
               '-vf', 'scale=-2:720,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[720]) + 'K',
               '"' + output_files['720'] + '"',
               '-vf', 'scale=-2:480,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[480]) + 'K',
               '"' + output_files['480'] + '"',
               '-vf', 'scale=-2:360,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[360]) + 'K',
               '"' + output_files['360'] + '"',
               '-vf', 'scale=-2:240,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[240]) + 'K',
               '"' + output_files['240'] + '"',
               '-vf', 'scale=-2:144,{}'.format(modifier), '-c:v', codec, '-b:v', str(bitrates[144]) + 'K',
               '"' + output_files['144'] + '"',
               ]
    return command


crete_folders()


def get_input_output_jobs():
    ffmpeg_jobs = []
    files = [f for f in listdir(arg_input_path) if isfile(join(arg_input_path, f)) and not f.startswith('.')]
    for file in files:
        bitrates = get_renditions(files_and_renditions[file.split('.mp4')[0]])
        full_input_file = join(arg_input_path, file)
        job_output_folders = {}
        for output_key, output_value in output_folders[selected_option].items():
            output_folder = join(arg_output_path, output_value)
            full_output_file = join(output_folder, file)
            job_output_folders[output_key] = full_output_file
        ffmpeg_jobs.append((full_input_file, codec_to_use, bitrates, job_output_folders))
    return ffmpeg_jobs


def worker(full_input_file, codec, bitrates, output_files):
    ffmpeg_command = []
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

    with multiprocessing.Pool() as pool:
        pool.starmap(worker, jobs)
