ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=11.2.0-cudnn8

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

LABEL maintainer ""

ENV DEBIAN_FRONTEND noninteractive

######################################
# install dependencies for vicos-demo (echolib and echocv)

ENV MODEL_DIR /opt

WORKDIR $MODEL_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Get echolib

COPY --from=echolib-base /usr/local/ /usr/local/

##################################
# install dependencies for deske-demo
RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools
RUN python3 -m pip install opencv-python>=4
RUN python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install future

COPY scripts ${MODEL_DIR}

#RUN wget https://box.vicos.si/vicos-cube/ploscice/docker/models/6_FINAL.pth ${MODEL_DIR}/model.pth
#RUN wget https://box.vicos.si/vicos-cube/ploscice/docker/models/6_seg_FINAL.pth ${MODEL_DIR}/model_seg.pth

RUN chmod +x ${MODEL_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py", "--model", "/opt/6_FINAL.pth", "--model_seg", "/opt/6_seg_FINAL.pth"]
