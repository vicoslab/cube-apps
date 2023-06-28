ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=11.2.0-cudnn8

FROM echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

LABEL maintainer "pm4824@student.uni-lj.si"

ENV DEBIAN_FRONTEND noninteractive

######################################
# install dependencies for vicos-demo (echolib and echocv)

ENV MODEL_DIR /opt

WORKDIR $MODEL_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Get echolib

COPY --from=echolib-base /usr/local/ /usr/local/

##################################
# install dependencies for poco-demo
RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools
RUN python3 -m pip install tensorflow==2.6.2 opencv-python>=4
RUN pip3 install future

COPY scripts ${MODEL_DIR}

RUN chmod +x ${MODEL_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py"]
