import argparse
import subprocess
from os import listdir, makedirs
from os.path import isfile, join, exists
import multiprocessing

parser = argparse.ArgumentParser(description='Generate renditions ')
parser.add_argument('-i', "--input", action='store', help='Folder where the renditions are', type=str,
                    required=True)
parser.add_argument('-o', "--output", action='store', help='Folder where the black and white renditions will be',
                    type=str, required=True)

args = parser.parse_args()

input_path = args.input
output_path = args.output

output_folders = {
    '1080p': '1080p_black_and_white',
    '720p': '720p_black_and_white',
    '480p': '480p_black_and_white',
    '360p': '360p_black_and_white',
    '240p': '240p_black_and_white',
    '144p': '144p_black_and_white',
}

input_folders = [
    '1080p',
    '720p',
    '480p',
    '360p',
    '240p',
    '144p',
]


def crete_folders():
    for key, value in output_folders.items():
        folder = output_path + '/' + value
        if not exists(folder):
            makedirs(folder)


def get_input_output_jobs():
    jobs = []
    for folder in input_folders:
        input_folder = join(input_path, folder)
        output_folder = join(output_path, output_folders[folder])
        files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and not f.startswith('.')]
        for file in files:
            full_input_file = join(input_folder, file)
            full_output_file = join(output_folder, file)
            jobs.append((full_input_file, full_output_file))
    return jobs


def format_command(full_input_file, full_output_file):
    print('processing {} {}'.format(full_input_file, full_output_file))
    command = ['ffmpeg', '-y', '-i', '"' + full_input_file + '"', '-vf', 'hue=s=0', '-c:a',
               'copy', '"' + full_output_file + '"'
               ]
    return command


def worker(full_input_file, full_output_file):
    try:
        ffmpeg_command = format_command(full_input_file, full_output_file)
        ffmpeg = subprocess.Popen(' '.join(ffmpeg_command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = ffmpeg.communicate()
    except Exception as e:
        print(full_input_file, full_output_file)
        print(e)


if __name__=="__main__":
    crete_folders()
    jobs = get_input_output_jobs()

    with multiprocessing.Pool() as pool:
        pool.starmap(worker, jobs)

