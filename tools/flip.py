import argparse
import subprocess
from os import makedirs
from utils import *

parser = argparse.ArgumentParser(description='Generate fliped renditions')
parser.add_argument('-i', '--input', action='store', help='Folder where the 1080 renditions are', type=str,
                    required=True)
parser.add_argument('-o', '--output', action='store', help='Folder where the fliped renditions will be',
                    type=str, required=True)

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

files = get_files(arg_input_path)

output_folders = {
    'vertical_flip': {
        '1080': '7.2',
        '720': '6.2',
        '480': '5.2',
        '360': '4.2',
        '240': '3.2',
        '144': '2.2',
    },
    'horizontal_flip': {
        '1080': '7.3',
        '720': '6.3',
        '480': '5.3',
        '360': '4.3',
        '240': '3.3',
        '144': '2.3',
    },
    'clockwise': {
        '1080': '7.4',
        '720': '6.4',
        '480': '5.4',
        '360': '4.4',
        '240': '3.4',
        '144': '2.4',
    },
    'counterclockwise': {
        '1080': '7.5',
        '720': '6.5',
        '480': '5.5',
        '360': '4.5',
        '240': '3.5',
        '144': '2.5',
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


files_and_renditions = get_files_and_reinditions('yt8m_data.csv')

selected_option = selected_bool_to_str()


def crete_folders():
    for key, value in output_folders[selected_option].items():
        folder = arg_output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def format_command(orig_file_name, codec, bitrate_1080, bitrate_720, bitrate_480, bitrate_360, bitrate_240, bitrate_144,
                   video_format, input_path, output_path):
    input_path_with_slash = input_path + '/'

    modifier = command_modifier[selected_option]

    command = ['ffmpeg', '-y',
               '-i', '"' + input_path_with_slash + orig_file_name + '"',
               '-vf', 'scale=-2:1080,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_1080 + 'K', '-f',
               video_format,
               '"' + output_path + '/' + output_folders[selected_option]['1080'] + '/{}'.format(orig_file_name + '"'),
               '-vf', 'scale=-2:720,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_720 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders[selected_option]['720'] + '/{}'.format(orig_file_name + '"'),
               '-vf', 'scale=-2:480,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_480 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders[selected_option]['480'] + '/{}'.format(orig_file_name + '"'),
               '-vf', 'scale=-2:360,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_360 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders[selected_option]['360'] + '/{}'.format(orig_file_name + '"'),
               '-vf', 'scale=-2:240,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_240 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders[selected_option]['240'] + '/{}'.format(orig_file_name + '"'),
               '-vf', 'scale=-2:144,{}'.format(modifier), '-c:v', codec, '-b:v', bitrate_144 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders[selected_option]['144'] + '/{}'.format(orig_file_name + '"'),
               ]
    print(' '.join(command))
    return command


crete_folders()

for file in files:
    file_name = file.split('.mp4')[0]
    bitrates = get_renditions(files_and_renditions[file_name])
    try:
        ffmpeg_command = format_command(file, 'libx264', str(bitrates[1080]), str(bitrates[720]), str(bitrates[480]),
                                        str(bitrates[360]), str(bitrates[240]), str(bitrates[144]), 'mp4',
                                        arg_input_path,
                                        arg_output_path)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
    except Exception as e:
        print(file)
        print(e)
