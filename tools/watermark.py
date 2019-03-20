import argparse
import csv
import subprocess
from os import listdir, makedirs
from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='Generate renditions with watermarks')
parser.add_argument('-i', "--input", action='store', help='Folder where the 1080 renditions are', type=str,
                    required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the renditions with watermarks will be',
                    type=str, required=True)

args = parser.parse_args()

input_path = args.input
output_path = args.output

files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]

output_folders = {
    '1080': '1080p_watermark',
    '720': '720p_watermark',
    '480': '480p_watermark',
    '360': '360p_watermark',
    '240': '240p_watermark',
    '144': '144p_watermark',
}

files_and_renditions = {}

with open('yt8m_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        files_and_renditions[row[3]] = row[5]


def get_renditions(renditions):
    resolutions = {144, 240, 360, 480, 720, 1080}
    ids = {160, 133, 134, 135, 136, 137}
    ladder = {}
    renditions = renditions.replace('[', '')
    renditions = renditions.replace(']', '')
    renditions = renditions.replace('}', '')
    renditions = renditions.replace('{', '')
    renditions = renditions.replace("'", '')
    renditions = renditions.strip()
    data = renditions.split(',')
    for step in data:
        try:
            id = int(step.split('-')[0].strip())
            resolution = int(step.split('x')[1].split(' ')[0])
            bitrate = float(step.split(':')[1])

            if resolution in resolutions and id in ids:
                ladder[resolution] = bitrate
        except:
            print('There was an error')
    return ladder


def crete_folders():
    for key, value in output_folders.items():
        folder = output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def format_command(orig_file_name, codec, bitrate_1080, bitrate_720, bitrate_480, bitrate_360, bitrate_240, bitrate_144,
                   video_format, input_path, output_path):
    input_path_with_slash = input_path + '/'

    command = ['ffmpeg', '-y', '-i', '"' + input_path_with_slash + orig_file_name + '"', '-i',
               '"' + './watermark/livepeer.png' + '"',
               '-filter_complex',
               '"[0:v]overlay=10:10,split=6[in1][in2][in3][in4][in5][in6];'
               '[in1]scale=-2:1080[out1];[in2]scale=-2:720[out2];[in3]scale=-2:480[out3];[in4]scale=-2:360[out4];'
               '[in5]scale=-2:240[out5];[in6]scale=-2:144[out6]"',
               '-map', '"[out1]"', '-c:v', codec, '-b:v', bitrate_1080 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['1080'] + '/{}'.format(orig_file_name + '"'),
               '-map', '"[out2]"', '-c:v', codec, '-b:v', bitrate_720 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['720'] + '/{}'.format(orig_file_name + '"'),
               '-map', '"[out3]"', '-c:v', codec, '-b:v', bitrate_480 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['480'] + '/{}'.format(orig_file_name + '"'),
               '-map', '"[out4]"', '-c:v', codec, '-b:v', bitrate_360 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['360'] + '/{}'.format(orig_file_name + '"'),
               '-map', '"[out5]"', '-c:v', codec, '-b:v', bitrate_240 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['240'] + '/{}'.format(orig_file_name + '"'),
               '-map', '"[out6]"', '-c:v', codec, '-b:v', bitrate_144 + 'K', '-f', video_format,
               '"' + output_path + '/' + output_folders['144'] + '/{}'.format(orig_file_name + '"')
               ]
    print(' '.join(command))
    return command


crete_folders()

for file in files:
    file_name = file.split('.mp4')[0]
    bitrates = get_renditions(files_and_renditions[file_name])
    try:
        ffmpeg_command = format_command(file, 'libx264', str(bitrates[1080]), str(bitrates[720]), str(bitrates[480]),
                                        str(bitrates[360]), str(bitrates[240]), str(bitrates[144]), 'mp4', input_path,
                                        output_path)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
    except Exception as e:
        print(file)
        print(e)
