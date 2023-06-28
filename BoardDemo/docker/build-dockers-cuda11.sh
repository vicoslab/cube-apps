#!/bin/bash
set -e

docker build . -t deske-demo:ubuntu18.04-cuda11.2.0-cudnn8 \
                --build-arg CUDA_VERSION=11.2.0-cudnn8 \
                --build-arg UBUNTU_VERSION=18.04
