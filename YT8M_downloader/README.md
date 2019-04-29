# YT8M downloader

This folder contains scripts to process the YT8M dataset and create a new dataset to run the experiments. This folder
also contains an already processed sample in `yt8m_data.csv` 

The sequence goes as follows:

metadata_extractor.py -> cleaner.py -> downloader.py  

- ## metadata_extractor.py
 
This script takes one mandatory param (input) and one optional param (output).

The input parameter is the path to the folder where the `tfrecords` are. `tfrecord` files for YT8M dataset can be downloaded from 
the [YT8M page](https://research.google.com/youtube8m/download.html) and more information about how they are made can be found in 
the [github repo of the project](https://github.com/google/youtube-8m).

In brief, they are binaries used by TensorFlow where data containing different labels and rgb values and other metadata can be extracted.

The output parameter is optional and it is the name of the file .csv where the values extracted from the records processed are going to be stored. It is optional as it defaults to yt8m_data.csv in the folder where the script is.

```python
python3 metadata_extractor.py -i /path/to/folder/with/tfrecords
```

- ## cleaner.py

Once we have all the data in the csv, next step is to clean it, provided that many of the assets supplied by the tfrecords is in many cases useless. Namely, many of the assets do not have all resolutions available in Youtube (1080p, 720p, 480p, 360p, 240p, 144p). Also, it might happen that we would like to limit the number of assets in our dataset. The cleaner takes three parameters (input, output, number):

The input parameter is the file previously generated.

The output parameter is optional and it is the file where the records processed are going to be stored. It is optional 
as it defaults to yt8m_data.csv in the folder where the script is.

The number parameter is the number of rows we want to keep (i.e. number of assets we want to have available).

This script discards videos where the renditions  (`144, 240, 360, 480, 720 and 
1080`) are not available.


- ## downloader.py

When all data is clean, the last step is to download it. This script takes one param (output) which is the folder where the
videos are going to be stored. Inside this folder, two sub-folders are going to be created:
- `raw/1080p`
- `trim/1080p`

The `raw` folder is where the videos are going to be stored in temporarily. After a video is downloaded it is trimmed 
to 10 seconds from the middle of the video. The trimmed videos are stored in the `trim` folder.

After a video is trimmed the original is deleted.
