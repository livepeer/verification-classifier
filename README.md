This repo contains a Dockerfile to make easy the interaction with the experiments.

##Build the image
To build the image, we have to type the following in the same folder of the repo:
```
docker build -t epicjupiter:v1 .
```

This will create a image based on `jupyter/datascience-notebook` but adding the needed python dependencies.

## Run the image
To run the image, we have to type
```
docker run -p 8888:8888 --volume="$(pwd)":/home/jovyan/work/ epicjupiter:v1
```

This will run the image on the port 8888 and mounts a volume with the contents of this repo in the folder `/home/jovyan/work/` which is accesible by jupyter and all the notebooks here will be acesible.

## Note on notebooks

### Compare_videos notebook

This notebook expects the videos to be in the data folder with the following structure

```
data
├── 3
│   └── 01.mp4
├── 4
│   └── 01.mp4
├── 5
│   └── 01.mp4
├── 6
│   └── 01.mp4
└── 7
    └── 01.mp4
```

Where

7 is the folder for the 1080p rendition
6 is the folder for the 720p rendition
5 is the folder for the 480p rendition
4 is the folder for the 360p rendition
3 is the folder for the 240p rendition

The result will be in the output folder

### Metric analysis notebook

This notebook expect a file `metrics.csv` in the output folder