
"""
Cloud function trigger for Vimeo videos.
It makes an API call to collect a list of videos
with Creative Commons license that can be used
for research purposes.
"""
import os
import json
import vimeo
import youtube_dl
import requests

def get_video_data(page, per_page):
    """
    Function to make API call to Vimeo and filter
    out video IDs, with CC-BY license
    """
    with open('secrets/credentials.json') as json_file:
        credentials = json.load(json_file)
        token = credentials['token']

    vimeo_client = vimeo.VimeoClient(token=token,
                                     key='{client_id}',
                                     secret='{client_secret}',
                                     scope='video_files'
                                     )

    query = '/videos?filter=CC-BY&page={}&per_page={}'.format(page, per_page)
    vimeo_response = vimeo_client.get(query).json()
    vimeo_data = vimeo_response['data']
    file_output = open("resp_text.txt", "w")
    file_output.write(json.dumps(vimeo_data))
    file_output.close()

def get_video_sources(data_file):
    """
    Function to collect IDs of videos that both have
    Creative Commons and a resolution of 1080p from a
    given input data file
    """
    with open(data_file) as data:
        data = json.load(data)

    video_sources = []
    for video in data:
        if (int(video['height']) == 1080 and
                'nudity' not in video['content_rating']):
            try:    
                video_id = 'vimeo/{}'.format(video['uri'].split('/')[-1])
                video_link = video['link']
                duration = video['duration']
                bitrate, extension, playlist_url = get_metadata(video_link)
                if bitrate:
                    video_sources.append({'link' : video_link,
                                        'bitrate': bitrate,
                                        'video_id' : video_id,
                                        'playlist_url' : playlist_url,
                                        'duration': duration,
                                        'extension': extension})
            except:
                print('Failed at video:', video)
    return video_sources

def get_metadata(video_url):
    """
    Function to extract information about
    a video asset from a given url
    """

    options = {'format': 'bestvideo/best' # choice of quality

               }

    ydl = youtube_dl.YoutubeDL(options)

    with ydl:
        meta = ydl.extract_info(video_url,
                                download=False # We just want to extract the info
                                )
    bitrate = None
    extension = None
    for stream_format in meta['formats']:
        if ('tbr' in stream_format and
                stream_format['tbr'] is not None):
            if stream_format['height'] == 1080 and 'm3u8' in stream_format['protocol']:
                extension = stream_format['ext']
                bitrate = int(stream_format['tbr'])
                playlist_url = stream_format['url']
                print(bitrate)

    return bitrate, extension, playlist_url

def main():
    """
    Main function
    """
    video_sources = []
    for i in range(12, 100):
        print('Page:', i)
        get_video_data(page=i, per_page=100)
        video_sources.extend(get_video_sources('resp_text.txt'))

        file_output = open("video_ids.txt", "w")
        file_output.write(json.dumps(video_sources))
        file_output.close()

        with open("video_ids.txt") as video_sources:
            video_sources = json.load(video_sources)

        for video_source in video_sources:
            # Cloud function api-endpoint
            url = "https://us-central1-epiclabs.cloudfunctions.net/create_source_http"

            # defining a params dict for the parameters to be sent to the API
            params = video_source

            # sending get request and saving the response as response object
            response = requests.get(url=url, params=params)
            print(response)
        os.remove("video_ids.txt")
main()
