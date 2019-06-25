# CLI INSTRUCTIONS

## 1.- Build the image

To build the image, we have to run the following command line from the root of the project:

```
docker build -t verifier:v1 .
```

This will create a image based on `python3` but adding the needed python dependencies. This image 
contains OpenCV, numpy, pandas and sklearn, among others

## 2.- Run the image

To run the image, we have to type:

```
docker run --rm -it --volume="$(pwd)/data":/data --ipc=host verifier:v1
```

This will run the image and mount a volume with the contents of the folder /data from this repo in the folder 
`/data` of the Docker image. If you have your videos and their assets and attacks located elsewhere, it is recommended that you copy them in this structure for simplicity.

If you are using symbolic links to point the videos from the data folder to other folder, you need to mount the other folder to be visible in the cointainer.

For example if we have symbolic links in the `data` folder pointing to `/videos/` folder we need a new volume as follows:


## 3.- Usage

Once inside the Docker image, the python script has the following structure:

```
python3 scripts/cli.py path-to-original-asset --renditions path-to-rendition --renditions path-to-rendition ... --model_url url-to-model
```
Note that you can add as many --rendition (-r) as you want.

The model that is being used is a thresholding of the mean value of the gaussian time-series. This model has the following scores:

* TPR: 0.9942
* TNR: 0.8342
* F20: 0.9819

This model only uses information of the legit assets, making it attack-agnostic.
## 4.- Profiling

It is possible to do profiling of the processes involved in the computation of the video metrics and subsequent metrics needed for inference.
The profiling tool [py-spy](https://github.com/benfred/py-spy) has been included in the requirements.txt file and installed within the docker container. Access to the SYS_PTRACE variable needs to be granted when running it, though:

```
docker run --rm -it --volume="$(pwd)/data":/data --ipc=host --cap-add SYS_PTRACE verifier:v1
```

Then, simply add the py-spy command to run over the python script:

```
py-spy -- python3 scripts/cli.py path-to-original-asset --renditions path-to-rendition --renditions path-to-rendition ... --model_url url-to-model
```

Alternatively, one can enable the --do_profiling flag:

```
python3 scripts/cli.py path-to-original-asset --renditions path-to-rendition --renditions path-to-rendition ... --model_url url-to-model --do_profiling 1
```

For memory profiling the tool [memory_profiling](https://pypi.org/project/memory-profiler/) has been included in the requirements.txt. To use it, execute the script with ``` mprof run ```:

```
mprof run python3 scripts/cli.py path-to-original-asset --renditions path-to-rendition --renditions path-to-rendition ... --model_url url-to-model 
```

If a plot is required, it is possible to obtain by means of:

```
mprof plot -o plot.svg
```