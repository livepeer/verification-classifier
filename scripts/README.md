# Data generation scripts

This folder contains scripts used to generate video data from video assets.

It has three main steps:

* Raw video data generation 
* Standard metrics data generation
* Livepeer's research metrics data generation

## Raw video data generation

These scripts are used to generate new video data using the downloaded segments from YT8M's dataset subsets. They compose the bulk of simulated "attacks". It is important to check the YT8M_downloader first [here](../YT8M_downloader/README.md) to obtain the original segments.

Scripts hosted here are referenced on the [Tools.ipynb](https://github.com/livepeer/verification-classifier/blob/documentation_enhacements/data-analysis/notebooks/Tools.ipynb) notebook from the data-analysis/notebooks folder and can be run there. The data-analysis/notebooks folder contains a Dockerfile with all necessary settings.

First script should be the one that generates the different renditions based on the downloaded renditions.

### encode_renditions.py

This script takes an input folder containing 1080p renditions and will encode them with the params in the metadata leaving them in the specified output folder. 

A sample of usaage is:

```
python enconde_renditions.py -i /path/to/1080pRenditions -o /path/to/renditions -m /path/to/metadatafile 
```

After we have all renditions, we can use the other scripts. They are listed here without any special order


### black_and_white.py 

This script makes the videos black and white.

This script receives 2 parameters:
- The input path (-i or --input) which is the folder containing the renditions.
- The output path (-o or --output) which is the folder where the black and white videos are going to be stored.
A sample of usaage is:

```
python black_and_white.py -i /path/to/renditions -o /path/to/renditions
```

### chroma_subsampling.py

This script does a chroma subsampling of the video.

This script receives 4 parameters:
- The input path (-i or --input) which is the folder containing the 1080p renditions.
- The output path (-o or --output) which is the folder where the subsampled videos are going to be stored.
- The metadata (-m or --metadata) which is the file containing data about the videos.
- The chosen subsampling (-s --subsampling) which is the chroma subsampling to be applied to the video. this one must be a valid one for ffmpeg, which can be checked [here](https://trac.ffmpeg.org/wiki/Chroma%20Subsampling)


It is important to note that the subsampling is done from 1080p renditions as it is hard to find videos with a different color space other than yuv420p, so we take the 1080p and subsample in the lower resolution renditions.

A sample of usaage is:

```
python chroma_subsampling.py -i /path/to/1080pRenditions -o /path/to/renditions -m /path/to/metadatafile -s yuv422p
```

### flip.py

This script does a flip / rotation of the video. It receives 3 parameters:

- The input path (-i or --input) which is the folder containing the renditions.
- The output path (-o or --output) which is the folder where the flipped videos are going to be stored.
- The desired flip or rotation:
    -  -vf or --vflip for the vertical flip
    -  -hf or --hflip for the horizontal flip
    -  -cf or for the 90 degrees clockwise rotation
    -  -ccf for the 90 degrees counterclockwise rotation

A sample of usaage is:
```
python flip.py -i /path/to/renditions -o /path/to/renditions -vf
```

### low_bitrate.py

This script lower the bitrate of a video.  It receives 4 parameters:
- The input path (-i or --input) which is the folder containing the 1080p renditions are.
- The output path (-o or --output) which is the folder where the videos with low bitrate are going to be stored.
- The metadata (-m or --metadata) which is the file containing data about the videos, the most important is the needed bitrate to enconde the video.
- The chosen divisor for the bitrate (-d, --divisor) which is the divisoe to be applied to the video bitrate. It must be greater than 0

A sample of usaage is:

```
python low_bitrate.py -i /path/to/1080pRenditions -o /path/to/renditions -d 4
```

### vignette.py

This script performs a vignette in the video. It has 3 parameters (One is optional):
- The input path (-i or --input) which is the folder containing the renditions.
- The output path (-o or --output) which is the folder where the vignetted videos are going to be stored.
- The angle (-a or --angle) which is the angle of the vignette filter to be applied to the video. This param is optional and by default is [PI/5](https://ffmpeg.org/ffmpeg-filters.html#vignette-1)


A sample of usaage is:

```
python vignette.py -i /path/to/1080pRenditions -o /path/to/renditions
```

### watermark.py

This script puts a watermark in the video. It has 4 parameters:
- The input path (-i or --input) which is the folder containing 1080p.
- The output path (-o or --output) which is the folder where the videos with watermark are going to be stored.
- The metadata (-m or --metadata) which is the file containing data about the videos, the most important is the needed bitrate to enconde the video.
- The watermark file (-w --watermark) which is the file containing the image to be applied to the video.
- The position (-x or --pos_x) in x pixels from the left at which the watermark will be located.
- The position (-y or --pos_y) in y pixels from the bottom at which the watermark will be located.
- The suffix (-s or --suffix) to give to the folder where encoded files will be stored

```
python watermark.py -i /path/to/1080pRenditions -o /path/to/renditions -m /path/to/metadatafile -w /path/to/watermarkfile -x pos_x -y pos_y -s suffix
```


## Reprocess videos

In the event of a failed encoded video, we added to the scripts the ability of process just the failed videos.

To do that the scripts have a new parametter -r that receives a file containing a video name per line like:

fmOwlwRugjU.mp4
HZ_ZWbNTfWs.mp4

Bad encoded videos can be spoted in the compare videos notebook in the second step by replacing the actual code by:

```python
metrics_dict = {}
file = open('workfile','w') 
list = os.listdir(originals_path.format('1080p')) # dir is your directory path
number_assets = len(list)
print ('Number of assets: {}'.format(number_assets))
count = 0

for original_asset in glob.iglob(originals_path.format('1080p') + '**', recursive=False):
    count += 1
    if os.path.isfile(original_asset): # filter dirs
       
        print('Processing asset {} of {}: {}'.format(count, number_assets, original_asset))
        start_time = time.time()
        renditions_list = []

        for folder in renditions_folders:
            rendition_folder = originals_path.format(folder)
            renditions_list.append(rendition_folder + os.path.basename(original_asset))

        asset_processor = VideoAssetProcessor(original_asset, renditions_list, metrics_list, 1, False)

        asset_metrics_dict = asset_processor.process()

        dict_of_df = {k: pd.DataFrame(v) for k,v in asset_metrics_dict.items()}

        metrics_df = pd.concat(dict_of_df, axis=1).transpose().reset_index(inplace=False)
        metrics_df = metrics_df.rename(index=str, columns={"level_1": "frame_num", "level_0": "path"})

        renditions_dict = {}
        for rendition in renditions_list:
            try:
                rendition_dict = {}
                for metric in metrics_list:

                    original_df = metrics_df[metrics_df['path']==original_asset][metric]
                    original_df = original_df.reset_index(drop=True).transpose().dropna().astype(float)

                    rendition_df = metrics_df[metrics_df['path']==rendition][metric]
                    rendition_df = rendition_df.reset_index(drop=True).transpose().dropna().astype(float)

                    if  'temporal' in metric:
                        x_original = np.array(original_df[rendition_df.index].values)
                        x_rendition = np.array(rendition_df.values)

                        [[manhattan]] = 1/abs(1-distance.cdist(x_original.reshape(1,-1), x_rendition.reshape(1,-1), metric='cityblock'))

                        rendition_dict['{}-euclidean'.format(metric)] = distance.euclidean(x_original, x_rendition)
                        rendition_dict['{}-manhattan'.format(metric)] = manhattan
                        rendition_dict['{}-cosine'.format(metric)] = distance.cosine(x_original, x_rendition)
                        rendition_dict['{}-cross-correlation'.format(metric)] = np.correlate(x_original, x_rendition)
                        rendition_dict['{}-mean'.format(metric)] = np.mean(x_rendition)
                        rendition_dict['{}-max'.format(metric)] = np.max(x_rendition)
                        rendition_dict['{}-std'.format(metric)] = np.std(x_rendition)
                        rendition_dict['{}-series'.format(metric)] = x_rendition

                    else:
                        rendition_dict[metric] = rendition_df.mean()
                    rendition_dict['size'] = os.path.getsize(rendition)
                renditions_dict[rendition] = rendition_dict
            except:
                print('Unable to measure rendition:', rendition)
                
                file.write(rendition)

                
        metrics_dict[original_asset] = renditions_dict   

        elapsed_time = time.time() - start_time 
        print('Elapsed time:', elapsed_time)
        print('***************************')

file.close()     
``` 

This will create a file called `workfile` which can be feeded into `make_files_to_reprocess.py` which is going to split it in different files, one per attack and one for the original asset not processed as expected.

This files can be the input of the scripts

## Standard metrics data generation

In this folder resides the subfolder [shell](https://github.com/livepeer/verification-classifier/tree/neural_net/scripts/shell) containing shell scripts useful to compute different metrics throughout all the generated assets. It uses external libraries (ffmpeg and libav) that are compiled in the build step of the Docker container.


### evaluate-ms-ssim.sh

This script evaluates the ms-ssim (multiscale ssim) metric. It expects two parameters which are the folder containing the video renditions (attacks) and the folder where the output logs are going to be stored for later processing.

This script computes the psnr and ssim metrics of the original rendition against all the resolutions of:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a ms-ssim folder is going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.
Further forms of attack can be easily expanded by adding more lines to the bash file.

### evaluate-psnr-ssim.sh

This script evaluates the psnr and ssim metrics. It expect two parameters which are the folder containing the renditions and the folder where the output is going to be stored.

This script computes the psnr and ssim metrics of the original rendition against all the resolutions of:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a ssim and psnr folders are going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.
Further forms of attack can be easily expanded by adding more lines to the bash file.

### evaluate-vmaf.sh

This script evaluates the vmaf metric. It expect two parameters which are the folder containing the renditions and the folder where the output is going to be stored.

This script computes the psnr and ssim metrics of the original rendition against all the resolutions of:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a vmaf folder is going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.

## Livepeer's research metrics data generation

As part of the research involved in this project, a series of metrics have been developed to account for video features that enable their classification.
This process is documented in subsequent articles published [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea) and [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380).
These scripts are also need by the [CLI](https://github.com/livepeer/verification-classifier/tree/neural_net/cli) module as part of the inference process previous to classification.
