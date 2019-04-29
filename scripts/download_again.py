import argparse
import youtube_dl

parser = argparse.ArgumentParser(description='Download renditions again')

parser.add_argument('-i', '--input', action='store', help='Input file containing the files to download again', type=str,
                    required=True)
parser.add_argument('-o', '--output', action='store', help='Output fodler for the renditions', type=str, required=True)

args = parser.parse_args()
input_file = args.input
output_folder = args.output

YOUTUBE_URL = 'https://www.youtube.com/watch?v={}'


def get_urls_to_reprocess(file_with_errors):
    ids = set()
    with open(file_with_errors) as file_to_read:
        for line in file_to_read:
            id = line.split('.mp4')[0]
            ids.add(YOUTUBE_URL.format(id))
    return ids


def download(url):
    with youtube_dl.YoutubeDL({'format': '137',
                               'outtmpl': output_folder + '/%(id)s.%(ext)s',
                               'quiet': True,
                               'fragment-retries': 100,
                               'retries': 100}) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            fn = ydl.prepare_filename(info_dict)
            return fn
        except youtube_dl.utils.DownloadError as err:
            print('Error downloading {}'.format(url))
            print(str(err))
        except Exception as err:
            print('Unknown exception')
            print(str(err))


if __name__ == "__main__":
    urls_to_download = get_urls_to_reprocess(input_file)
    for url in urls_to_download:
        result = download(url)
        print('downloaded {}'.format(result))
