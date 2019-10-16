# Compile
FROM jupyter/datascience-notebook
COPY requirements.txt .
COPY ./scripts /scripts
RUN pip install -r requirements.txt
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    yasm \
    pkg-config \
    x264 \
    libx264-dev \
    zlib1g-dev \
    ninja-build \
    nasm && \
    rm -rf /var/lib/apt/lists/*
# RUN mkdir -p vmaf && cd vmaf && \
# curl -s https://codeload.github.com/Netflix/vmaf/tar.gz/v1.3.14 | tar zxvf - -C . && \
# cd vmaf-1.3.14 && \
# make && \
# make install
# RUN mkdir -p ffmpeg && cd ffmpeg && \
# curl -s http://ffmpeg.org/releases/ffmpeg-4.1.1.tar.gz | tar zxvf - -C . && \
# cd ffmpeg-4.1.1 && \
# ./configure \
# --enable-version3 --enable-gpl --enable-nonfree --enable-libx264 --enable-libvmaf --disable-debug && \
# make && \
# make install && \
# make distclean
# RUN git clone https://github.com/livepeer/TranscodingVerification.git && \
# cd TranscodingVerification && \
# meson .build && \
# ninja -C .build install
# RUN git clone https://github.com/lu-zero/libav.git && \
# cd libav && \
# git checkout mea && \
# ./configure --enable-libmea && \
# make && \
# make install
# Make jupyter available from external sources
RUN sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" /home/jovyan/.jupyter/jupyter_notebook_config.py
RUN sed -i "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" /home/jovyan/.jupyter/jupyter_notebook_config.py