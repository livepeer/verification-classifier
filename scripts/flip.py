import argparse
import multiprocessing
import subprocess
from os import listdir, makedirs
from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='Generate flipped renditions')
parser.add_argument('-i', '--input', action='store', help='Folder where the renditions are', type=str,
                    required=True)
parser.add_argument('-o', '--output', action='store', help='Folder where the fliped renditions will be',
                    type=str, required=True)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-vf', '--vflip', action='store_true', help='Flip video vertically')
group.add_argument('-hf', '--hflip', action='store_true', help='Flip video horizontally')
group.add_argument('-cf', '--clockflip', action='store_true', help='Rotate 90 degrees clockwise')
group.add_argument('-ccf', '--counterclockflip', action='store_true', help='Rotate 90 degrees counterclockwise')
parser.add_argument('-r', "--reprocess", action='store', help='Input file with files to reprocess', type=str,
                    required=False)

args = parser.parse_args()

arg_input_path = args.input
arg_output_path = args.output
vertical_flip = args.vflip
horizontal_flip = args.hflip
clockwise_flip = args.clockflip
counterclockwise_flip = args.counterclockflip

input_folders = [
    '1080p',
    '720p',
    '480p',
    '360p',
    '240p',
    '144p',
]

reprocess = False
file_to_reprocess = None

if args.reprocess is not None:
    reprocess = True
    file_to_reprocess = args.reprocess

output_folders = {
    'vertical_flip': {
        '1080p': '1080p_flip_vertical',
        '720p': '720p_flip_vertical',
        '480p': '480p_flip_vertical',
        '360p': '360p_flip_vertical',
        '240p': '240p_flip_vertical',
        '144p': '144p_flip_vertical',
    },
    'horizontal_flip': {
        '1080p': '1080p_flip_horizontal',
        '720p': '720p_flip_horizontal',
        '480p': '480p_flip_horizontal',
        '360p': '360p_flip_horizontal',
        '240p': '240p_flip_horizontal',
        '144p': '144p_flip_horizontal',
    },
    'clockwise': {
        '1080p': '1080p_rotate_90_clockwise',
        '720p': '720p_rotate_90_clockwise',
        '480p': '480p_rotate_90_clockwise',
        '360p': '360p_rotate_90_clockwise',
        '240p': '240p_rotate_90_clockwise',
        '144p': '144p_rotate_90_clockwise',
    },
    'counterclockwise': {
        '1080p': '1080p_rotate_90_counterclockwise',
        '720p': '720p_rotate_90_counterclockwise',
        '480p': '480p_rotate_90_counterclockwise',
        '360p': '360p_rotate_90_counterclockwise',
        '240p': '240p_rotate_90_counterclockwise',
        '144p': '144p_rotate_90_counterclockwise',
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


selected_option = selected_bool_to_str()

cpu_count = multiprocessing.cpu_count()
cpu_to_use = 1 if reprocess else int(round(cpu_count / len(output_folders[selected_option])))


def crete_folders():
    for key, value in output_folders[selected_option].items():
        folder = arg_output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def format_command(full_input_file, full_output_file):
    print('processing input {}'.format(full_input_file))
    modifier = command_modifier[selected_option]

    command = ['ffmpeg', '-y', '-i', '"' + full_input_file + '"', '-vf', modifier, '"' + full_output_file + '"']
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
    print('{} files to reprocess in {}'.format(len(file_list), input_path))
    return file_list


def get_input_output_jobs():
    ffmpeg_jobs = []
    for input_folder in input_folders:
        job_input_folder = join(arg_input_path, input_folder)
        job_output_folder = join(arg_output_path, output_folders[selected_option][input_folder])

        if reprocess:
            files = get_files_from_file(job_input_folder, file_to_reprocess)
        else:
            files = [f for f in listdir(job_input_folder) if
                     isfile(join(job_input_folder, f)) and not f.startswith('.')]
        for file in files:
            full_input_file = join(job_input_folder, file)
            full_output_file = join(job_output_folder, file)
            ffmpeg_jobs.append((full_input_file, full_output_file))
    return ffmpeg_jobs


def worker(full_input_file, full_output_file):
    ffmpeg_command = []
    out = None
    err = None
    try:
        ffmpeg_command = format_command(full_input_file, full_output_file)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
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
