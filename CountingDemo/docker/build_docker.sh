#!/bin/bash
set -e

docker build . -t counting-demo:ubuntu20.04-cuda11.8.0-cudnn8 \
                --build-arg CUDA_VERSION=11.8.0-cudnn8 \
                --build-arg UBUNTU_VERSION=20.04