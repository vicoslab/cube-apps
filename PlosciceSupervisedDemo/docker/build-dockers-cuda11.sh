#!/bin/bash
set -e

if [[ ! -e scripts/model_A012_fold0.pth ]]; then
    echo "Not building - missing model file."
    exit 1
fi
docker build . -t ploscice-supervised-demo:ubuntu18.04-cuda11.2.0-cudnn8 \
                --build-arg CUDA_VERSION=11.2.0-cudnn8 \
                --build-arg UBUNTU_VERSION=18.04
