# Context

This contribution involves research and attempt to tackle the problem of verifying that 
the transcoded content itself is a reasonable match for the original source given a good-faith effort at transcoding.

Eventually, the mission consists on developing a verification classifier that will give a pass / fail output (and with what confidence score), for a given segment of a given asset.

A series of articles on the topic can be found [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea).

This repo contains a Dockerfile to enable the interaction with the experiments and the necessary instructions.

## 1.- Build the image
To build the image, we have to type the following in the same folder of the repo:
```
docker build -t epicjupiter:v1 .
```

This will create a image based on `jupyter/datascience-notebook` but adding the needed python dependencies. This image 
contains ffmpeg with VMAF and libav with ms-ssim.

## 2.- Run the image
To run the image, we have to type:
```
docker run -p 8888:8888 --volume="$(pwd)":/home/jovyan/work/ epicjupiter:v1
```

This will run the image on the port 8888 and mount a volume with the contents of this repo in the folder 
`/home/jovyan/work/`.

Copy, paste (and ammend by removing spurious information) the URL provided in the console and navigate to the work folder to access the notebooks.
Alternatively, navigate to http://127.0.0.1:8888 and copy / paste the provided token in the console into the Password or token input box to log in.

## 3.- Notebooks

The notebooks used in the experiments are inside the folder work/notebooks

### 3.1.- Tools.ipynb

In order to run experiments there is need for preparing some data.

This notebook contains different sections to generate different datasets or auxiliary files that are needed in order to run the other notebooks.

The notebook can be found [here](notebooks/Tools.ipynb)


### 3.2.- Compare_videos.ipynb

We have taken a number of assets from Youtube’s YT8M dataset and encoded a few renditions from there. Specifically, we have taken about 140 videos from this dataset, established the 1080p rendition as original, and encoded 10 seconds of each to 720p, 480p, 360p and 240p. For the sake of simplicity, we have reduced the respective bitrates to be equal to those used by YouTube for each rendition (you can find a more detailed article on how this can be done here).

We have also invited a few more full reference metrics to the party, namely cosine, euclidean and hamming distances, so we add more diversity to the analysis.

Once we have gathered our renditions, we have iterated video by video (4 renditions x 140 videos = 560 specimens) and extracted their mean PSNR, SSIM, MS-SSIM, VMAF, cosine, Hamming and euclidean hash distances with respect to the original 1080p rendition.

This notebook expects the videos to reside in a data folder with the following structure. The structure needs to be created in the data folder of the repo, which is empty as default.

```
data
├── 1080p
│   └── 01.mp4
├── 720p
│   └── 01.mp4
├── 480p
│   └── 01.mp4
├── 360p
│   └── 01.mp4
│── 240p
│   └── 01.mp4
└── 144p
    └── 01.mp4    
```

The result will be stored in the /output folder

The notebook can be found [here](notebooks/Compare_videos.ipynb)

### 3.3.- Metric analysis.ipynb

This notebook expects a file `metrics.csv` in the output folder.

The notebook can be found [here](notebooks/Metric_analysis.ipynb)