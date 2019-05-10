#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import tensorflow as tf
import os
import sys
from urllib.request import urlopen
import youtube_dl

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
output_file = CURRENT_PATH + '/' + 'yt8m_data.csv'

parser = argparse.ArgumentParser(description='Make dataset from tfrecords')
parser.add_argument('-i', '--input', action='store', help='Folder where the tfrecords are', type=str, required=True)
parser.add_argument('-o', '--output', action='store', help='File where the result is going to be sotred', type=str,
                    required=False)

args = parser.parse_args()

input_path = args.input

if args.output is not None:
    output_file = args.output


def get_all_tfrecords(tfrecords_path):
    tfrecords_to_process = [os.path.join(tfrecords_path, f) for f in os.listdir(tfrecords_path) if
                            os.path.isfile(os.path.join(tfrecords_path, f)) and f.endswith('.tfrecord')]
    return tfrecords_to_process


# This function collects the data provided by youtube-dl, such as rendition tables, number of views, etc.
def get_metadata(video_id: str) -> str or None:
    url = 'https://www.youtube.com/watch?v=' + video_id
    ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'})
    try:
        with ydl:
            result = ydl.extract_info(url, download=False)
            return result
    except youtube_dl.utils.DownloadError:
        return None


# For privacy reasons the video IDs in the dataset were provided with a codification.
# Instructions and further information are available here:
#      https://research.google.com/youtube8m/video_id_conversion.html
def get_real_id(random_id: str) -> str:
    url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(random_id[0:2], random_id)
    request = urlopen(url).read()
    real_id = request.decode()
    return real_id[real_id.find(',') + 2:real_id.find(')') - 1]


def save_data_to_csv(data_to_save, file_to_write):
    data_to_save.to_csv(file_to_write)


def process_records(records):
    vid_ids = []

    data = pd.DataFrame()

    for record in records:

        # Iterate the contents of the TensorFlow record
        for example in tf.python_io.tf_record_iterator(record):

            # A TensoFlow Example is a mostly-normalized data format for storing data for
            # training and inference.  It contains a key-value store (features); where
            # each key (string) maps to a Feature message (which is oneof packed BytesList,
            # FloatList, or Int64List). Features for this data set are:
            #     -id
            #     -labels
            #     -mean_audio
            #     -mean_rgb
            tf_example = tf.train.Example.FromString(example)

            # Once we have the structured data, we can extract the relevant features (id and labels)
            vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            pseudo_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            labels = tf_example.features.feature['labels'].int64_list.value

            # The id provided from the TensoFlow example needs some processing in order to build a valid link to a
            # YouTube video
            try:
                real_id = get_real_id(pseudo_id)

                # Get the youtube-dl valuable metadata
                data_video = get_metadata(real_id)

                if data_video:

                    # We are interested in expanding the labels information with features such as title,
                    # creator, number of views and duration
                    title = data_video['title']
                    creator = data_video['creator']
                    view_count = data_video['view_count']
                    duration = data_video['duration']

                    # youtube-dl library supplies data regarding formats mixed for both audio and video.
                    # We are only interested in mp4 inputs, so we need to separate
                    formats_dict = []
                    for format_type in data_video['formats']:
                        try:
                            if format_type['ext'] == 'mp4':
                                formats_dict.append({format_type['format']: format_type['tbr']})
                        except:
                            e = sys.exc_info()
                            print('Exception ', e)

                    # Collect the data in the dataframe
                    data = data.append({'id': real_id,
                                        'ladder': formats_dict,
                                        'title': title,
                                        'creator': creator,
                                        'views': view_count,
                                        'duration': duration,
                                        'labels': labels},
                                       ignore_index=True)
            except:
                e = sys.exc_info()
                print('Exception ', e)
    return data


if __name__ == "__main__":
    tfrecords = get_all_tfrecords(input_path)
    processed_data = process_records(tfrecords)
    save_data_to_csv(processed_data, output_file)
