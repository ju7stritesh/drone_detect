FROM tensorflow/tensorflow:1.13.1-gpu-py3

MAINTAINER Chip Garner <chip@garnertotalenergy.com>

RUN pip3 install numpy && \
    pip3 install Keras==2.2.4 && \
    pip3 install keras-resnet==0.2.0 && \
    pip3 install boto3 && \
    pip3 install pillow

RUN mkdir OpenCV && cd OpenCV

RUN apt-get update && apt-get install -y \
  build-essential \
  checkinstall \
  cmake \
  pkg-config \
  yasm \
  libtiff5-dev \
  libjpeg-dev \
  libjasper-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libdc1394-22-dev \
  libxine2-dev \
  libv4l-dev \
  libtbb-dev \
  libeigen3-dev \
  libqt4-dev \
  libgtk2.0-dev \
  libmp3lame-dev \
  libopencore-amrnb-dev \
  libopencore-amrwb-dev \
  libtheora-dev \
  libvorbis-dev \
  libxvidcore-dev \
  x264 \
  v4l-utils \
  libgtk2.0-dev \
  unzip \
  wget

RUN cd /opt && \
  wget https://github.com/Itseez/opencv/archive/4.1.0.zip -O opencv-4.1.0.zip -nv && \
  unzip opencv-4.1.0.zip && \
  cd opencv-4.1.0 && \
  rm -rf build && \
  mkdir build && \
  cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
  make -j8 && \
  make install && \
  ldconfig

