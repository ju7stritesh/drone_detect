#!/usr/bin/env bash

sudo nvidia-docker run --privileged -it \
    -v /home/sesha/Documents/drone_detect:/dev/projects \
    drone_detect_image bash