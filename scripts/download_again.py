import argparse
import datetime
import os
import subprocess
import youtube_dl

parser = argparse.ArgumentParser(description='Download renditions again')

parser.add_argument('-i', '--input', action='store', help='Input file containing the files to download again', type=str,
                    required=True)
parser.add_argument('-o', '--output', action='store', help='Output fodler for the renditions', type=str, required=True)

args = parser.parse_args()
input_file = args.input
output_folder = args.output

YOUTUBE_URL = 'https://www.youtube.com/watch?v={}'


def format_ffmpeg_command(full_input_file, full_output_file, start_time, end_time):
    command = ['ffmpeg', '-y',
               '-i', '"' + full_input_file + '"', '-ss', start_time, '-to', end_time, full_output_file
               ]
    return command


def get_urls_to_reprocess(file_with_errors):
    ids = set()
    with open(file_with_errors) as file_to_read:
        for line in file_to_read:
            id = line.split('.mp4')[0]
            ids.add(YOUTUBE_URL.format(id))
    return ids


def download(url):
    with youtube_dl.YoutubeDL({'format': '137',
                               'outtmpl': output_folder + '/%(id)s.%(ext)s' + '_tmp',
                               'quiet': True,
                               'fragment-retries': 100,
                               'retries': 100}) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            fn = ydl.prepare_filename(info_dict)
            return fn, info_dict['duration']
        except youtube_dl.utils.DownloadError as err:
            print('Error downloading {}'.format(url))
            print(str(err))
        except Exception as err:
            print('Unknown exception')
            print(str(err))


def get_start_end_time(video_duration):
    half_duration = int(video_duration / 2)
    start_time = str(datetime.timedelta(seconds=half_duration))
    end_time = str(datetime.timedelta(seconds=(half_duration + 10)))
    return start_time, end_time


def trim_file(downloaded_file):
    start_time, end_time = get_start_end_time(downloaded_file[1])
    output_file = downloaded_file[0].split('_tmp')[0]
    ffmpeg_command = format_ffmpeg_command(downloaded_file[0], output_file, start_time, end_time)
    ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = ffmpeg.communicate()
    status = ffmpeg.wait()
    os.remove(downloaded_file[0])
    print('FFMPEG exit status {}'.format(status))
    return output_file


if __name__ == "__main__":
    urls_to_download = get_urls_to_reprocess(input_file)
    for url in urls_to_download:
        result = download(url)
        print('downloaded {}'.format(result))
        trimed_file = trim_file(result)
        print('trimed {}'.format(trimed_file))