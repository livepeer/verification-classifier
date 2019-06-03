import csv
from os import listdir
from os.path import isfile, join, exists
from typing import List, Dict


def get_files(input_path: str) -> List[str]:
    return [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]


def get_files_and_renditions(input_csv_file: str) -> Dict:
    files_and_renditions = {}
    with open(input_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            files_and_renditions[row[3]] = row[5]
    return files_and_renditions


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
            id = int(step.split('-')[0].strip())
            resolution = int(step.split('x')[1].split(' ')[0])
            bitrate = float(step.split(':')[1])

            if resolution in resolutions and id in ids:
                ladder[resolution] = bitrate
        except:
            print('There was an error')
    return ladder

