ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.7.0-cudnn8

FROM echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive

# Get echolib
COPY --from=echolib-base /usr/local/ /usr/local/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
RUN pip install future opencv_python torch==1.13.1 torchvision==0.14.1 pandas scikit-learn scikit-image tensorboard matplotlib segmentation_models_pytorch scipy tqdm

ENV DIR /opt
WORKDIR $DIR

ARG CEDIRNET_VERSION=12
RUN git clone --branch vicos_demo --depth 1 https://DockerPull:glpat-scfQzKpmYhGz1wi3nXDD@gitlab.fri.uni-lj.si/vicos/divid/cedirnet-dev.git
RUN cd cedirnet-dev

# install models 
ARG MODEL_VERSION="no_depth_large"
RUN wget https://box.vicos.si/vicos-cube/ClothDemo/docker/${MODEL_VERSION}.pth /opt/model.pth

# install inference code
COPY scripts/run_main.py /opt/run_main.py
COPY scripts/echolib_wrapper.py /opt/echolib_wrapper.py
COPY scripts/configs/${MODEL_VERSION}.py /opt/model_args.py

# add cedirnet code to python path
ENV PYTHONPATH "${PYTHONPATH}:/opt/cedirnet-dev/src"

ENTRYPOINT ["python3", "/opt/run_main.py" , "--model", "/opt/model.pth"]

