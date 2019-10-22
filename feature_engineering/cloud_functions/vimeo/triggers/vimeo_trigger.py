
"""
Cloud function trigger for Vimeo videos.
It makes an API call to collect a list of videos
with Creative Commons license that can be used
for research purposes.
"""
import json
import vimeo
import youtube_dl

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

def get_video_ids(data_file):
    """
    Function to collect IDs of videos that both have
    Creative Commons and a resolution of 1080p from a
    given input data file
    """
    with open(data_file) as data:
        data = json.load(data)

    video_ids = []
    for video in data:
        if (int(video['height']) == 1080 and
                'nudity' not in video['content_rating']):
            video_id = video['uri'].split('/')[-1]
            video_link = video['link']
            duration = video['duration']
            bitrate, extension, playlist_url = get_metadata(video_link)
            if bitrate:
                video_ids.append({'link' : video_link,
                                  'bitrate': bitrate,
                                  'video_id' : video_id,
                                  'playlist_url' : playlist_url,
                                  'duration': duration,
                                  'extension': extension})
    return video_ids

def get_metadata(video_url):
    """
    Function to extract information about
    a video asset from a given url
    """

    options = {'format': 'bestvideo/best', # choice of quality
               'extractaudio' : False,      # only keep the audio
               'outtmpl': '%(id)s',        # name the file the ID of the video
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
    video_ids = []
    for i in range(1, 3):
        print('Page:', i)
        get_video_data(page=i, per_page=100)
        video_ids.extend(get_video_ids('resp_text.txt'))

    file_output = open("video_ids.txt", "w")
    file_output.write(json.dumps(video_ids))
    file_output.close()
main()
