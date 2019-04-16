# YT8M downloader

This folder contains scripts to process the YT8M dataset and create a new dataset to run the experimients. This folder
Also contains an already processed sample in `yt8m_data.csv` 

The secuence is:

metadata_extractor.py -> cleaner.py -> downloader.py  

- ## metadata_extractor.py
 
This script takes one mandatory param (input) and one optional param (output).

The input parameter is the folder where the `tfrecords` are. A `tfrecord` can be downloaded from 
the [YT8M page](https://research.google.com/youtube8m/download.html) and more information can be found in 
the [github repo of the project](https://github.com/google/youtube-8m)

The output parameter is optional and it is the file where the records processed are going to be stored. It is optional 
as it defaults to yt8m_data.csv in the folder where the script is.

```python
python3 metadata_extractor.py -i /path/to/folder/with/tfrecords
```

- ## cleaner.py

Once we have all the data in the csv, next step is clean it. The cleaner takes three parameters (input, output, number):

The input parameter is the file previously generated.

The output parameter is optional and it is the file where the records processed are going to be stored. It is optional 
as it defaults to yt8m_data.csv in the folder where the script is.

The number parameter is the number of rows we want to keep.

This script discards videos with a different number of renditions of the wanted ones (`144, 240, 360, 480, 720 and 
1080`)  


- ## downloader.py

When all data is clean, the last step is download it. This script takes one param (output) which is the folder where the
videos are going to be stored. Inside this output, two folders are going to be created:
- `raw/1080p`
- `trim/1080p`

The `raw` folder is where the videos are going to be stored in a temporal way. After a video is downloaded it is timmed 
to 10 seconds from the middle of the video. The trimmed videos are stored in the `trim` folder.

After a video is trimmed the original is deleted.
