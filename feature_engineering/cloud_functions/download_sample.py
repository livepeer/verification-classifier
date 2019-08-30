import os
import random


files_to_download = 30
dataset_path = '../../stream/'
originals_bucket = 'gs://livepeer-verifier-originals/'
renditions_bucket = 'gs://livepeer-verifier-renditions/'
original = '1080p/'

resolutions = ['144p', '240p', '360p', '480p', '720p', '1080p']

attacks = [
            'watermark/',
            'watermark-345x114/',
            'watermark-856x856/',
            'vignette/',
            'low_bitrate_4/',
            'low_bitrate_8/',
            'black_and_white/',
            # 'flip_vertical/',
            'rotate_90_clockwise/'
          ]


original_list = os.popen('gsutil ls {}'.format(originals_bucket)).read().split('\n')
original_list = [asset[asset.rfind('/') + 1:] for asset in original_list[:-1]]
print('There is a total of {} original files in the bucket'.format(len(original_list)))

total = 0

random.shuffle(original_list)

for file in original_list:
    print('----- Checking for file: {} -----'.format(file))
    found = True
    renditions_list = [res + '/' for res in resolutions[:-1]]
    attack_list = [res + '_' + attk for res in resolutions for attk in attacks]
    renditions_list += attack_list

    for rend in renditions_list:
        not_exists = os.popen('gsutil -q stat {}; echo $?'.format(renditions_bucket + rend + file)).read()
        if not_exists == '1\n':
            found = False
            print('{} was not found\n'.format(renditions_bucket + rend + file))
            break

    if found:
        print('Downloading {}\n'.format(file))
        os.system('gsutil cp {} {}'.format(originals_bucket + file, dataset_path + original))

        for rend in renditions_list:
            os.system('gsutil cp {} {}'.format(renditions_bucket + rend + file, dataset_path + rend))

        total += 1

    if total >= files_to_download:
        print('Finishing downloading')
        break
