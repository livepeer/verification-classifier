# Context

This contribution involves research and attempts to tackle the problem of verifying that the transcoded content itself is a reasonable match for the original source given a good-faith effort at transcoding.

The mission consists on developing a verification classifier that will give a pass / fail output (and with what confidence score), for a given segment of a given asset's rendition.

A series of articles on the topic can be found [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea) and [here](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380).

This folder contains a Dockerfile to enable the interaction with a CLI for computing an asset's renditions Euclidean distance values.
Further insight about how this works can be gained by interacting with the [data-analysis](https://github.com/livepeer/verification-classifier/tree/documentation_enhacements/data-analysis) section and reading the aforementioned publications. 
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

Section 1 of [Tools.ipynb](https://github.com/livepeer/verification-classifier/blob/documentation_enhacements/data-analysis/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.


# 2.2. Metrics computation with external tools

There are different standard metrics (VMAF, MS-SSIM, SSIM and PSNR) provided by external tools (ffmpeg and libav) which can be run from the data-analysis/notebooks folder Tools.ipynb notebook. The notebook provides info on how to use them, but also inside the scripts folder [here](/scripts/README.md)

Section 2 of [Tools.ipynb](https://github.com/livepeer/verification-classifier/blob/documentation_enhacements/data-analysis/notebooks/Tools.ipynb) notebook helps in the usage in case a notebook is preferred as a means of interaction.

Alternatively, the scripts can be run separately as bash scripts.

# 2.3. Data analysis with jupyter notebooks

At this step we should have the required data in the form of video assets and attacks as well as the metrics extracted with the external tools which may be required by some of the notebooks.

Further information about this notebooks can be found [here](data-analysis/README.md)

# 3. Command Line Interface: cli

Once models are trained and available, a command line interface to interact with them and obtain predictions is offered in the cli folder.

# 4. Common usage scripts: scripts

Several utility scripts are hosted in this folder for convenience. They are needed at different stages of the process and for different Docker instances.
