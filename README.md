# Context

This contribution involves research and attempts to tackle the problem of verifying that 
the transcoded content itself is a reasonable match for the original source given a good-faith effort at transcoding.

The mission consists on developing a verification classifier that will give a pass / fail output (and with what confidence score), for a given segment of a given asset's rendition.

A series of articles on the topic can be found [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea).

This folder contains a Dockerfile to enable the interaction with a CLI for computing an asset's renditions Euclidean distance values.
Further insight about how this works can be gained by interacting with the data-analysis section and reading the aforementioned publications.

## 1.- Build the image
To build the image, we have to run the following command line:

```
docker build -t verifier:v1 .
```

This will create a image based on `python3` but adding the needed python dependencies. This image 
contains OpenCV, numpy, pandas and sklearn, among others

## 2.- Run the image
To run the image, we have to type:

```
docker run --rm -it --volume="$(pwd)/data-analysis/data":/data-analysis/data --ipc=host verifier:v1
```

This will run the image and mount a volume with the contents of the folder data-analysis/data from this repo in the folder 
`/data-analysis/data` of the Docker image. If you have your videos and their assets located elsewhere, it is recommended that you 
copy them in this structure for simplicity.

If you are using symbolic links to point the videos from the data folder to other folder, you need to mount the other folder to be visible in the cointainer.

For example if we have symbolic links in the `data` folder pointing to `/videos/` folder we need a new volume as follows:


## 3.- Usage
Once inside the Docker image, the python script has the following structure:

```
python3 src/cli.py path-to-original-asset --renditions path-to-rendition --renditions path-to-rendition ... -metrics implemented-metric
```
Note that you can add as many --rendition (-r) and --metrics (-m) arguments as you want.
Metrics can be one of:

- temporal_canny
- temporal_difference
- temporal_psnr
- temporal_mse
- histogram_distance
- hash_euclidean
- hash_hamming
- hash_cosine

