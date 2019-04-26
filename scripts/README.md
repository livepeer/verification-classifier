# Data generation

This folder contains scripts in order to generate new data using the downloaded data from YT8M. So it is important check the YT8M_downloader first [here](../YT8M_downloader/README.md).

This scripts are referenced on the Tools.ipynb notebook from the data-analysis/notebooks folder and can be run there. In the data-analysis/notebooks folder it is a Dockerfile containing all necesary in order to run this scripts.

First script should be the one that generate the different renditions based on the downloaded renditions.

## encode_renditions.py

This script takes an input folder containing 1080p renditions and will encode them with the params in the metadata leaving them in the specified output folder. 

A sample of usaage is:

```
python enconde_renditions.py -i /path/to/1080pRenditions -o /path/to/renditions -m /path/to/metadatafile 
```

After we have all renditions, we can use the other scripts. They are listed here without any special order


## black_and_white.py 

This script makes the videos black and white.

This script receives 2 parameters:
- The input path (-i or --input) which is the folder containing the renditions.
- The output path (-o or --output) which is the folder where the black and white videos are going to be stored.
A sample of usaage is:

```
python black_and_white.py -i /path/to/renditions -o /path/to/renditions
```

## chroma_subsampling.py

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

## flip.py

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

## low_bitrate.py

This script lower the bitrate of a video.  It receives 4 parameters:
- The input path (-i or --input) which is the folder containing the 1080p renditions are.
- The output path (-o or --output) which is the folder where the videos with low bitrate are going to be stored.
- The metadata (-m or --metadata) which is the file containing data about the videos, the most important is the needed bitrate to enconde the video.
- The chosen divisor for the bitrate (-d, --divisor) which is the divisoe to be applied to the video bitrate. It must be greater than 0

A sample of usaage is:

```
python low_bitrate.py -i /path/to/1080pRenditions -o /path/to/renditions -d 4
```

## vignette.py

This script performs a vignette in the video. It has 3 parameters (One is optional):
- The input path (-i or --input) which is the folder containing the renditions.
- The output path (-o or --output) which is the folder where the vignetted videos are going to be stored.
- The angle (-a or --angle) which is the angle of the vignette filter to be applied to the video. This param is optional and by default is [PI/5](https://ffmpeg.org/ffmpeg-filters.html#vignette-1)


A sample of usaage is:

```
python vignette.py -i /path/to/1080pRenditions -o /path/to/renditions
```

## watermark.py

This script puts a watermark in the video. It has 4 parameters:
- The input path (-i or --input) which is the folder containing 1080p.
- The output path (-o or --output) which is the folder where the videos with watermark are going to be stored.
- The metadata (-m or --metadata) which is the file containing data about the videos, the most important is the needed bitrate to enconde the video.
- The watermark file (-w --watermark) which is the file containing the image to be applied to the video.

```
python watermark.py -i /path/to/1080pRenditions -o /path/to/renditions -m /path/to/metadatafile -w /path/to/watermarkfile
```


# Data analysis with external tools

In this folder, it is a subfolder shell containing shell scripts in order to compute different metrics 


## evaluate-ms-ssim.sh

This script evaluates the ms ssim (multiscale ssim) metric. It expect two parameters which are the folder containing the renditions and the folder where the output is going to be stored.

This script compute the ms ssim metric of the original rendition against:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a ms-ssim folder is going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.

## evaluate-psnr-ssim.sh

This script evaluates the psnr and ssim metrics. It expect two parameters which are the folder containing the renditions and the folder where the output is going to be stored.

This script compute the psnr and ssim metrics of the original rendition against:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a ssim and psnr folders are going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.


## evaluate-vmaf.sh

This script evaluates the vmaf metric. It expect two parameters which are the folder containing the renditions and the folder where the output is going to be stored.

This script compute the vmaf metric of the original rendition against:

- watermark
- flip vertical
- flip horizontal
- rotate 90 clockwise
- rotate 90 counterclockwise

In the output folder a vmaf folder is going to be created containing one subfolder per resolution and attack to compare to. Inside those folders there will be a file per rendition containing the result.