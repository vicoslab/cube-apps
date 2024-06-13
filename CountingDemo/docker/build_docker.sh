#!/bin/bash
set -e

docker build . -t counting-demo:ubuntu20.04-cuda12.5.0 \
                --build-arg CUDA_VERSION=12.5.0 \
                --build-arg UBUNTU_VERSION=20.04