This repo contains a Dockerfile to make easy the interaction with the experiments.

## Build the image
To build the image, we have to type the following in the same folder of the repo:
```
docker build -t epicjupiter:v1 .
```

This will create a image based on `jupyter/datascience-notebook` but adding the needed python dependencies. This image 
contains ffmpeg with VMAF and libav with ms-ssim.

## Run the image
To run the image, we have to type
```
docker run -p 8888:8888 --volume="$(pwd)":/home/jovyan/work/ epicjupiter:v1
```

This will run the image on the port 8888 and mounts a volume with the contents of this repo in the folder 
`/home/jovyan/work/` which is accesible by jupyter and all the notebooks here will be acesible.

## Note on notebooks


The notebooks are inside of work/ntoebooks

### Compare_videos notebook

The notebook can be found [here](notebooks/Compare_videos.ipynb)

This notebook expects the videos to be in the data folder with the following structure

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

The result will be in the output folder

### Metric analysis notebook

The notebook can be found [here](notebooks/Metric analysis.ipynb)

This notebook expects a file `metrics.csv` in the output folder

### Tools notebook

The notebook can be found [here](notebooks/Tools.ipynb)

This notebook contains different sections to generate different datasets or auxiliary files in order to run the other notebooks
