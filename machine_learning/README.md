# Data Analysis Jupyter notebooks

Jupyter notebooks employed in the experiments are stored here. To interact with them, it is reccommended to build and launch
Docker image as explained below.

## 1.- Build the image
To build the image, we have to run the following shell script that lies in the same folder of the repo:
```
bash build_docker.sh
```

This will create a image based on `jupyter/datascience-notebook` but adding the needed python dependencies. 

## 2.- Run the image
To run the image, we have to type the following IN THE ROOT FOLDER OF THE REPO:
```
docker run -p 8888:8888 --volume="$(pwd)":/home/jovyan/work/ epicjupiter-ml:v1
```

This will run the image on the port 8888 and mount a volume with the contents of this repo in the folder 
`/home/jovyan/work/`.

Copy, paste (and ammend by removing spurious information) the URL provided in the console and navigate to the work folder to access the notebooks.
Alternatively, navigate to http://127.0.0.1:8888 and copy / paste the provided token in the console into the Password or token input box to log in.

If you are using symbolic links to point the videos from the data folder to other folder, you need to mount the other folder to be visible in the cointainer.

For example if we have symbolic links in the `data` folder pointing to `/videos/` folder we need a new volume as follows:

```
docker run -d -p 8888:8888 --volume="$(pwd)":/home/jovyan/work/ --volume=/videos/:/videos/ epicjupiter:v1
```

Also it is important to have read and write permissions in the output folder in order to be able to store the results.

## 3.- Notebooks

The notebooks used in the experiments are inside the folder work/notebooks. 
The training data is obtained from the .csv files dropped in data_analytics/output folder.

### 3.1.- Supervised_predictive_model.ipynb

This notebook enables the training of different models under supervised machine learning techniques (i.e. requiring a ground truth to learn from).
Techniques explored are:

* Neural network with Keras
* Random Forest
* AdaBoost
* Support Vector Machine
* XGBoost

The notebook can be found [here](notebooks/Supervised_predictive_model.ipynb)


### 3.2.- Unsupervised_predictive_model.ipynb

This notebook enables the training of different models under unsupervised machine learning techniques (i.e. without need for labeled data).
Techniques explored are:

* One Class SVM
* Isolation Forest
* Autoencoder

The notebook can be found [here](notebooks/Unsupervised_predictive_model.ipynb)

