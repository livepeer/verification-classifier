# Context

This contribution involves research and attempts to tackle the problem of verifying that the transcoded content itself is a reasonable match for the original source given a good-faith effort at transcoding.

The mission consists on developing a verification classifier that will give a pass / fail output (and with what confidence score), for a given segment of a given asset's rendition.

A series of articles on the topic can be found [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea) and [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380).

This folder contains a Dockerfile to enable the interaction with a CLI for computing an asset's renditions Euclidean distance values.
Further insight about how this works can be gained by interacting with the [feature_engineering](feature_engineering) section and reading the aforementioned publications. 
Full documentation on the cli can be found on the cli folder of this repo [here](cli/README.md).

This repo contains several folders to separate different steps of the data generation and analysis.

# 1. Bulk video data generation: YT8M_Downloader

We are using 10 second chunks of videos from the YouTube-8M Dataset available [here](https://research.google.com/youtube8m/).
Previous work with this dataset can be found [here](https://github.com/epiclabs-io/YT8M).

All the information and the scripts to create the assets reside inside the [YT8M_downloader](YTM8_downloader) folder and are explained in [this](YT8M_downloader/README.md) document.

# 2. Video data analysis: data_analytics

From the raw video dataset created we obtain different features out of the analysis made with different tools.

## 2.1. Generation of renditions
As part of the feature extraction, we want to generate different variations of the videos including different renditions, flipped videos, etc. Some of these variations constitute the bulk of what we label as "attacks". Other constitute "good" renditions where no distortions are included.

To obtain the different "attacks", we provide several scripts in order to perform each variation.

All the information and the scripts can be found inside the scripts folder [here](scripts/README.md)

Section 1 of [Tools.ipynb](feature_engineering/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.


# 2.2. Metrics computation with external tools

There are different standard metrics (VMAF, MS-SSIM, SSIM and PSNR) provided by external tools (ffmpeg and libav) which can be run from the data-analysis/notebooks folder Tools.ipynb notebook. The notebook provides info on how to use them, but also inside the scripts folder [here](/scripts/README.md)

Section 2 of [Tools.ipynb](feature_engineering/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.

Alternatively, the scripts can be run separately as bash scripts.

# 2.3. Data analysis with jupyter notebooks

At this step we should have the required data in the form of video assets and attacks as well as the metrics extracted with the external tools which may be required by some of the notebooks.

Further information about this notebooks can be found [here](feature_engineering/README.md)

# 3. Interfaces: CLI and API

Once models are trained and available, a [CLI](https://github.com/livepeer/verification-classifier/tree/master/cli) and a [RESTful API](https://github.com/livepeer/verification-classifier/tree/master/api) to interact with them and obtain predictions are made available.
The bash scripts launch_cli.sh and launch_api.sh can be run from the root folder of the project.

# 4. Common usage scripts: scripts

Several utility scripts are hosted in this folder for convenience. They are needed at different stages of the process and for different Docker instances.

# 5. Unit Tests

Unit tests are located in testing/tests folder. Some tests are using data included in repository (under testing/tests/data, machine_learning/output/models, etc.), while other require the following assets to be downloaded and extracted into ../data directory:
1. [Dataset CSV](https://storage.cloud.google.com/feature_dataset/yt8m-large.tar.gz)
2. [YT8M renditions mini dataset](https://storage.cloud.google.com/feature_dataset/renditions-mini.tar) 
3. [Small dataset for CI/CD](https://storage.cloud.google.com/feature_dataset/renditions-nano.tar.gz)

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
