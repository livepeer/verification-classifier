#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
from typing import Dict

EXPECTED_RESOLUTIONS = 6
WANTED_RESOLUTIONS = {144, 240, 360, 480, 720, 1080}
WANTED_RESOLUTION_IDS = {160, 133, 134, 135, 136, 137}
OUTPUT_FILE = 'yt8m_data.csv'
HEADER = ['', 'creator', 'duration', 'id', 'labels', 'ladder', 'title', 'views']

parser = argparse.ArgumentParser(description='Clean data')
parser.add_argument('-i', '--input', action='store', help='Input file to clean', type=str, required=True)
parser.add_argument('-o', '--output', action='store', help='Output file to store results', type=str, required=False)
parser.add_argument('-n', '--number', action='store', help='Number of records to keep', type=int, required=True)

args = parser.parse_args()

input_file = args.input
number_records_to_keep = args.number

if args.output:
    OUTPUT_FILE = args.output


def get_renditions(renditions: str) -> Dict:
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

            if resolution in WANTED_RESOLUTIONS and resolution_id in WANTED_RESOLUTION_IDS:
                ladder[resolution] = bitrate
        except:
            print('There was an error')
    return ladder


def read_all_rows(file_to_read):
    processed_rows = 0
    read_rows = []
    with open(file_to_read) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        next(csv_reader)
        for row in csv_reader:
            if processed_rows < number_records_to_keep:
                renditions = get_renditions(row[5])
                if EXPECTED_RESOLUTIONS == len(renditions):
                    processed_rows += 1
                    read_rows.append(row)
            else:
                break
    return read_rows


def write_all_rows(rows_to_write):
    with open(OUTPUT_FILE, mode='w') as file_to_write:
        csv_writer = csv.writer(file_to_write, delimiter=',', quotechar='"')
        csv_writer.writerow(HEADER)
        for row in rows_to_write:
            csv_writer.writerow(row)


if __name__ == "__main__":
    all_rows = read_all_rows(input_file)
    write_all_rows(all_rows)
