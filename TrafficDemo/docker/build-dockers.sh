#!/bin/bash
set -e

# build original detectron docker from GitHub dockerfile
docker build https://github.com/skokec/detectron-traffic-signs.git#villard:docker/detectron_cuda11 \
                    -t detectron-traffic-signs:ubuntu16.04-cuda11.1.1-cudnn8  \
                    --build-arg CUDA_VERSION=11.1.1-cudnn8 \
                    --build-arg UBUNTU_VERSION=16.04

# ResNet50
docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda11.1.1-cudnn8-resnet50 \
                        --build-arg UBUNTU_VERSION=16.04 \
                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda11.1.1-cudnn8 \
                        --build-arg BACKBONE=resnet50

# ResNet101
#docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda11.1.1-cudnn8-resnet101 \
#                        --build-arg UBUNTU_VERSION=16.04 \
#                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda11.1.1-cudnn8 \
#                        --build-arg BACKBONE=resnet101
