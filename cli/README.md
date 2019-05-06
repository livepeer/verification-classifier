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

For the models, you can try one of any [from the official repository of the verifier](https://github.com/livepeer/verification-classifier/raw/master/machine_learning/output/models):
* Random Forest: https://github.com/livepeer/verification-classifier/raw/master/machine_learning/output/models/random_forest.pickle.dat
* XGBoost: https://github.com/livepeer/verification-classifier/raw/master/machine_learning/output/models/XGBoost.pickle.dat
* AdaBoost: https://github.com/livepeer/verification-classifier/raw/master/machine_learning/output/models/AdaBoost.pickle.dat

This list is subject to extension.