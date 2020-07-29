# Context

This contribution involves research and attempts to tackle the problem of verifying that the transcoded content itself is a reasonable match for the original source given a good-faith effort at transcoding.

The mission consists on developing a verification classifier that will give a pass / fail output (and with what confidence score), for a given segment of a given asset's rendition.

A series of articles on the topic can be found [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea) and [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380).

An up-to-date verifier implementation is exposed through the [API](api/README.md). The implementation section below documents its design. 

# Implementation
This section is intended to give readers a high level understanding of currently implemented verification process, without diving into too much details.
## Interface
REST API is the main interface for video verification. It is implemented with Flask and runs on a Gunicorn server inside Docker container. The API is documented [here](api/README.md). 
## Verification process
Verification process consist of following steps:
### 1. Preparation
Source video and each rendition video are made accessible through file system.
### 2. Pre-verification
Metadata attributes, such as width, height and framerate, are read from video file headers and compared among source video, renditions and assumed values passed in the API call. Handled by [Verifier](verifier/verifier.py) class.
### 3. Frame matching
The frame matching algorithm goal is to choose closest by presentation time stamp frame pairs from source and rendition videos. Once PTS is extracted, the task is trivial, if source and rendition FPS are same. If frame rates doesn't match, the algorithm works as follows: 
1. An excessive number of frames is uniformly sampled from source video. The number is determined as MAX(N_SAMPLES, N_SAMPLES * MAX{SOURCE_FPS/RENDITION_FPS}). This allows to increase probability of finding best matching timestamps in case rendition FPS is lower than source fps. 
2. Presentation timestamps of rendition video frames are iterated to find closest matching frame for each master frame. If the timestamp difference for a given pair exceeds 1/(2*SOURCE_FPS), the pair is discarded.
3. Resulting set of frame pairs returned for metrics computation.  

Implemented in [VideoAssetProcessor](scripts/asset_processor/video_asset_processor.py) class. 
### 4. Metrics computation
On a rendition video level, following numerical metrics are computed:
- size_dimension_ratio  

For each frame pair, following numerical metrics are computed:
- temporal_dct
- temporal_gaussian_mse
- temporal_gaussian_difference
- temporal_threshold_gaussian_difference
- temporal_histogram_distance

One important thing to note regarding frame-level metrics, is that all of them, except temporal_histogram_difference, are applied to V channel of HSV-converted frame image. Without full-channel metrics, it would be trivial for an attacker to craft a very obviously tampered video, which would pass the verification.
     
The code for metric computation is located [here](scripts/asset_processor/video_metrics.py).
### 5. Metrics aggregation
Each per-frame pair metric is aggregated across frame pairs to get a single value for source-rendition pair in question. Currently, the aggregation function is a simple mean for each metric.
### 6. Metrics scaling
The final step is to scale metrics according to video resolution. After that, we have features which could be used with models.
### 7. Classification
The process of determining whether the video is tampered is viewed as a binary classification task. The Positive class or 1 is assigned to tampered videos, while Negative (0) designates untampered renditions, which accurately represent the source video.  
Once features are extracted for select source-rendition video pair, they are fed to following models:
- One Class Support Vector Machine  
This is an anomaly detection model, it was fit to untampered renditions to learn the 'normal' distribution of features and detect outliers. It is characterized by lower number of False Positives, but is somewhat less sensitive to tampered videos. Being unsupervised model, it is expected to generalize well on novel data.
- CatBoost binary classifier.  
This supervised model is trained on a labeled dataset and typically achieve higher accuracy, than OCSVM model.
### 8. Meta-model application
To make a final prediction, the following rule is applied to classification models output:
- if OCSVM prediction is "Untampered", return "Untampered"
- otherwise, return CatBoost model prediction

The goal is to reduce the number of False Positives to prevent wrongfully penalizing transcoder nodes. OCSVM model is expected to have higher precision (low FP) on novel data. If OCSVM predicts the observation is an inlier, we'll go with it, otherwise we'll use supervised model output. 

# Repository structure
## 1. Bulk video data generation: YT8M_Downloader

We are using 10 second chunks of videos from the YouTube-8M Dataset available [here](https://research.google.com/youtube8m/).
Previous work with this dataset can be found [here](https://github.com/epiclabs-io/YT8M).

All the information and the scripts to create the assets reside inside the [YT8M_downloader](YTM8_downloader) folder and are explained in [this](YT8M_downloader/README.md) document.

## 2. Video data analysis: data_analytics

From the raw video dataset created we obtain different features out of the analysis made with different tools.

### 2.1. Generation of renditions
As part of the feature extraction, we want to generate different variations of the videos including different renditions, flipped videos, etc. Some of these variations constitute the bulk of what we label as "attacks". Other constitute "good" renditions where no distortions are included.

To obtain the different "attacks", we provide several scripts in order to perform each variation.

All the information and the scripts can be found inside the scripts folder [here](scripts/README.md)

Section 1 of [Tools.ipynb](feature_engineering/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.


### 2.2. Metrics computation with external tools

There are different standard metrics (VMAF, MS-SSIM, SSIM and PSNR) provided by external tools (ffmpeg and libav) which can be run from the data-analysis/notebooks folder Tools.ipynb notebook. The notebook provides info on how to use them, but also inside the scripts folder [here](/scripts/README.md)

Section 2 of [Tools.ipynb](feature_engineering/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.

Alternatively, the scripts can be run separately as bash scripts.

### 2.3. Data analysis with jupyter notebooks

At this step we should have the required data in the form of video assets and attacks as well as the metrics extracted with the external tools which may be required by some of the notebooks.

Further information about this notebooks can be found [here](feature_engineering/README.md)

## 3. Interfaces: CLI and API

Once models are trained and available, a [CLI](https://github.com/livepeer/verification-classifier/tree/master/cli) and a [RESTful API](https://github.com/livepeer/verification-classifier/tree/master/api) to interact with them and obtain predictions are made available.
The bash scripts launch_cli.sh and launch_api.sh can be run from the root folder of the project.

## 4. Common usage scripts: scripts

Several utility scripts are hosted in this folder for convenience. They are needed at different stages of the process and for different Docker instances.

## 5. Unit Tests

Unit tests are located in testing/tests folder. Some tests are using data included in repository (under testing/tests/data, machine_learning/output/models, etc.), while other require the following assets to be downloaded and extracted into ../data directory:
1. [Dataset CSV](https://storage.cloud.google.com/feature_dataset/yt8m-large.tar.gz)
2. [YT8M renditions mini dataset](https://storage.cloud.google.com/feature_dataset/renditions-mini.tar)  

To run tests:
- Install prerequisites
```
sudo apt install ffmpeg
pip install -r requirements.txt
```
- Run tests
```
python -m pytest testing/tests
``` 
