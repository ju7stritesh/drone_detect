This code is borrowed from Keras Implementation of this model at https://github.com/fizyr/keras-retinanet and updated to run on Stanford Drone Data Set

## Installation

1) Clone this repository.  
2) Ensure numpy is installed using `pip install numpy --user`  
3) In the repository, execute `pip install . --user`.  
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.  
5) Optionally, install `pycocotools` by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.  


## Inference

1) Run inference.py (access the drone videos from "videos" folder)

## Instructions to run different urls

1) Build the docker image using docker build -t drone_image_detect .  
2) Run ./RunContainer.sh (In the file change the path to your local path)  
3) Run cd /dev/projects  
3) Run python AccessUrl.py <url> <frames per second> for RTSP/RTMP stream  
4) Run python TCPImageReceiver.py <ip address> <Port number> <frames per second> for TCP stream (IP address 0.0.0.0)

Medium article - https://ritesh-4165.medium.com/drones-with-artificial-intelligence-will-soon-become-a-powerful-tool-a-new-perspective-86f5e7e6f888
