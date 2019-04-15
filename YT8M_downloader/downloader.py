#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import datetime
import multiprocessing
import os
import subprocess
from typing import Dict
import youtube_dl

parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument('-o', '--output', action='store', help='Folder where the videos will be', type=str, required=True)

args = parser.parse_args()
output_folder = args.output + '/raw/1080p'
output_trim_folder = args.output + '/trim/1080p'
output_trim_folder_placeholder = args.output + '/trim/1080p/{}'
EXPECTED_RESOLUTIONS = 6
YOUTUBE_URL = 'https://www.youtube.com/watch?v={}'
current_path = os.path.dirname(os.path.abspath(__file__))


def save_dict_to_file(dic):
    with open('yt8m_data_processed.txt', 'w') as dict_file:
        dict_file.write(str(dic))


def load_dict_from_file():
    with open('yt8m_data_processed.txt', 'r') as dict_file:
        data = dict_file.read()
    return eval(data)


def download(url):
    try:
        ydl_video = youtube_dl.YoutubeDL({'format': '137',
                                          'outtmpl': output_folder + '/%(id)s.%(ext)s'})

        info_dict = ydl_video.extract_info(url, download=True)
        fn = ydl_video.prepare_filename(info_dict)
        return fn, info_dict['duration']

    except youtube_dl.utils.DownloadError:
        return None


def get_renditions(renditions: str) -> Dict:
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
            resolution_id = int(step.split('-')[0].strip())
            resolution = int(step.split('x')[1].split(' ')[0])
            bitrate = float(step.split(':')[1])

            if resolution in resolutions and resolution_id in ids:
                ladder[resolution] = bitrate
        except:
            print('There was an error')
    return ladder


def read_data():
    read_rows = []
    read_ids = []
    with open(current_path + '/' + 'yt8m_data.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            renditions = get_renditions(row[5])
            if EXPECTED_RESOLUTIONS == len(renditions):
                read_rows.append((row[3], renditions))
                read_ids.append((row[3],))
    return read_rows, read_ids


def format_ffmpeg_command(full_input_file, full_output_file, start_time, end_time):
    command = ['ffmpeg', '-y',
               '-i', '"' + full_input_file + '"', '-ss', start_time, '-to', end_time, full_output_file
               ]
    return command


def worker(url_to_download):
    full_input_path, duration = download(url_to_download)
    start_time, end_time = get_start_end_time(duration)
    filename = full_input_path.split('/')[-1]
    full_output_path = output_trim_folder_placeholder.format(filename)
    ffmpeg_command = []
    out = None
    err = None
    try:
        ffmpeg_command = format_ffmpeg_command(full_input_path, full_output_path, start_time, end_time)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
    except Exception as e:
        print('Error processing ', full_input_path)
        print('The error was ', e)
        print('Executing ', ffmpeg_command)
        print('Out ', out)
        print('Error ', err)


def get_start_end_time(video_duration):
    half_duration = int(video_duration / 2)
    start_time = str(datetime.timedelta(seconds=half_duration))
    end_time = str(datetime.timedelta(seconds=(half_duration + 10)))
    return start_time, end_time


if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_trim_folder):
        os.makedirs(output_trim_folder)

    all_rows, ids_to_process = read_data()

    with multiprocessing.Pool() as pool:
        pool.starmap(worker, ids_to_process)
