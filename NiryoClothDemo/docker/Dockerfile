ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.7.1-cudnn8

# Base echolib
FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib-base

# Main image with CUDA
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Copy echolib from base image
COPY --from=echolib-base /usr/local/ /usr/local/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3-dev python3-numpy-dev python3-pip \
    libopencv-dev libgl1 libglib2.0-0 software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.6 and create a virtual environment for it
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.6 python3.6-venv python3.6-dev iputils-ping iproute2

# Install Python 3.6 dependencies
RUN python3.6 -m pip install numpy==1.19.5 pyniryo==1.1.1 opencv-python==4.3.0.38

# Install dependencies in system Python (default version)
RUN pip install --upgrade pip 
RUN pip install future opencv_python torch==1.13.1 torchvision==0.14.1 \
    pandas scikit-learn scikit-image tensorboard matplotlib \
    segmentation_models_pytorch scipy tqdm numpy

# Set working directory
ENV MODEL_DIR=/opt
WORKDIR ${MODEL_DIR}

# Clone cedirnet repository
ARG CEDIRNET_VERSION=12
RUN git clone --branch vicos_demo --depth 1 https://DockerPull:gldt-aGCecL9t-EEznPDtQEaj@gitlab.fri.uni-lj.si/vicos/divid/cedirnet-dev.git cedirnet-dev
RUN cd cedirnet-dev

# Copy models
ARG MODEL_VERSION="no_depth_large"
COPY ${MODEL_VERSION}.pth /opt/model.pth
COPY center_model.pth /opt/center_model.pth

# Copy inference code
COPY scripts/*.py ${MODEL_DIR}
COPY scripts/config/${MODEL_VERSION}.py ${MODEL_DIR}/model_args.py
COPY scripts/niryoOne.py ${MODEL_DIR}/niryoOne.py

# Add cedirnet to Python path
ENV PYTHONPATH="${PYTHONPATH}:${MODEL_DIR}/cedirnet-dev/src"

# Run both scripts using different Python versions
ENTRYPOINT ["sh", "-c", "python3.6 /opt/niryoOne.py & python3 /opt/intermediate_connector.py & python3 /opt/run_main.py --model /opt/model.pth --center_model /opt/center_model.pth & wait"]