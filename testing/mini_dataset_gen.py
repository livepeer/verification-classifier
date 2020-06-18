"""
This script generates a subset of rendition and source videos arranged in directory structure
"""
import argparse
import os
import pathlib
import tqdm
import shutil
import logging

# init logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def main(args):
    # create output folder
    os.makedirs(args.output, exist_ok=True)
    rend_files = list(pathlib.Path(args.input).glob('**/*.*'))
    rend_type_count = {}
    master_copy_jobs = []
    for f in tqdm.tqdm(rend_files, desc='Copying renditions...'):
        rendition_id = os.sep.join(str(f).split(os.sep)[-2:])
        master_id = f'1080p{os.sep}{str(f).split(os.sep)[-1]}'
        master_path = args.input + os.sep + master_id
        target_master_path = args.output + os.sep + master_id
        target_path = args.output + os.sep + rendition_id
        target_dir = os.sep.join(target_path.split(os.sep)[:-1])
        rendition_type = str(f).split(os.sep)[-2]
        if master_id == rendition_id:
            continue
        if not os.path.exists(master_path):
            logger.warning(f'Source video doesn\'t exist: {master_path}')
            continue
        if rend_type_count.get(rendition_type, 0) >= args.count:
            continue
        if not os.path.exists(target_path):
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(f, target_path)
        if rendition_type not in rend_type_count:
            rend_type_count[rendition_type] = 0
        rend_type_count[rendition_type] += 1
        master_copy_jobs.append((master_path, target_master_path))
    # copy master videos of renditions
    for src, dst in tqdm.tqdm(master_copy_jobs, desc='Copying source videos...'):
        if not os.path.exists(dst):
            os.makedirs(os.sep.join(dst.split(os.sep)[:-1]), exist_ok=True)
            shutil.copy(src, dst)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--count', help='Max number of videos of each rendition type to retain', default=3)
    ap.add_argument('-i', '--input', help='Input dir', default='../../dataset')
    ap.add_argument('-o', '--output', help='Output dir', default='../../dataset-mini/')
    args = ap.parse_args()
    main(args)
