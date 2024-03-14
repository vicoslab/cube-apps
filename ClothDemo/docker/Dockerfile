ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.7.1-cudnn8

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive

# Get echolib
COPY --from=echolib-base /usr/local/ /usr/local/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
RUN pip install future opencv_python torch==1.13.1 torchvision==0.14.1 pandas scikit-learn scikit-image tensorboard matplotlib segmentation_models_pytorch scipy tqdm

ENV MODEL_DIR /opt
WORKDIR ${MODEL_DIR}

ARG CEDIRNET_VERSION=12
RUN git clone --branch vicos_demo --depth 1 https://DockerPull:glpat-La3echzmAvsRyxBhzNWz@gitlab.fri.uni-lj.si/vicos/divid/cedirnet-dev.git
RUN cd cedirnet-dev

# install models 
ARG MODEL_VERSION="no_depth_large"
COPY ${MODEL_VERSION}.pth /opt/model.pth
COPY center_model.pth /opt/center_model.pth

# install inference code
COPY scripts/*.py ${MODEL_DIR}
COPY scripts/config/${MODEL_VERSION}.py ${MODEL_DIR}/model_args.py


# add cedirnet code to python path
ENV PYTHONPATH "${PYTHONPATH}:${MODEL_DIR}/cedirnet-dev/src"

ENTRYPOINT ["python3", "/opt/run_main.py" , "--model", "/opt/model.pth", "--center_model", "/opt/center_model.pth"]