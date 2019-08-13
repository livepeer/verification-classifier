# CLI INSTRUCTIONS

## 1.- Build the image and create a container

To build the image and create a container, we have to run the following bash script located in the root of the project:

```
./launch_cli.sh
```

This will create a Docker image based on `python3` and adds the needed python dependencies.

This will run the image and mount a volume with the contents of the folder /data from this repo in the folder
`/data` of the Docker image. 
If you have your videos and their assets and attacks located elsewhere, it is recommended that you copy them in this structure for simplicity.

## 2.- Usage

Once inside the Docker image, the python script has the following structure:

```
python3 scripts/cli.py path-to-original-asset url-to-model --renditions path-to-rendition --renditions path-to-rendition
```
Note that you can add as many --rendition (-r) as you want.

The model that is being used is a One-Class Support Vector Machine maintained by Livepeer. For other models feature compatibility is not guaranteed.
If the folder /tmp/model is not found in the Docker image, it is created and the model is downloaded from [here](https://storage.googleapis.com/verification-models/verification.tar.gz).

This model only uses information of the legit assets, making it attack-agnostic.

## 3.- Profiling

It is possible to do profiling of the processes involved in the computation of the video metrics and subsequent metrics needed for inference.
The profiling tool [py-spy](https://github.com/benfred/py-spy) has been included in the requirements.txt file and installed within the docker container. Access to the SYS_PTRACE variable needs to be granted when running it, though:

```
docker run --rm -f Dockerfile-cli -it --volume="$(pwd)/stream":/stream --ipc=host --cap-add SYS_PTRACE verifier-cli:v1
```

Then, simply add the py-spy command to run over the python script:

```
py-spy -- python3 scripts/cli.py path-to-original-asset url-to-model --renditions path-to-rendition --renditions path-to-rendition ...
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