"""
Module for installation of ffmpeg
It requires a valid binary version of ffmpeg and ffprobe to be
located in the ffmpeg_binaries folder within the imports folder
"""

import subprocess

def install():
    """
    Library for collecting and making ffmpeg and ffprobe available for the cloud function
    """
    # print('Creating FFMPEG and FFPROBE folders in bin')
    bash_command = "mkdir /usr/local/bin/ffmpeg"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # print(output, error)
    # print('Moving FFMPEG and FFPROBE binaries to bin')
    bash_command = "mv /imports/ffmpeg_binaries/ffmpeg /usr/local/bin/ffmpeg && mv /imports/ffmpeg_binaries/ffprobe /usr/local/bin/ffmpeg"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # print(output, error)
    # print('Creating symlinks')
    bash_command = "ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg && ln -s /usr/local/bin/ffmpeg/ffprobe /usr/bin/ffprobe"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # print(output, error)
    