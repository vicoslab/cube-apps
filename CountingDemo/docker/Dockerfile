ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.8.0-cudnn8

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

#LABEL maintainer ""

ENV DEBIAN_FRONTEND noninteractive

######################################
# install dependencies for vicos-demo (echolib and echocv)

ENV MODEL_DIR /opt
WORKDIR $MODEL_DIR
RUN apt-get update && apt-get install -y --no-install-recommends && apt-get install -y wget \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Get echolib
COPY --from=echolib-base /usr/local/ /usr/local/

##################################
# install dependencies for count
RUN wget https://box.vicos.si/vicos-cube/count/DAVE/verification.pth
RUN wget https://box.vicos.si/vicos-cube/count/DAVE/DAVE_0_shot.pth
RUN wget https://box.vicos.si/vicos-cube/count/DAVE/img.png

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install numpy
RUN python3 -m pip install scikit-image
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install opencv-python>=4
RUN python3 -m pip install matplotlib

# dependency for echolib
RUN python3 -m pip  install future

COPY scripts ${MODEL_DIR}

RUN chmod +x ${MODEL_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py","--zero_shot","--skip_train","--model_name", "DAVE_0_shot","--model_path", "/opt","--backbone", "resnet50","--swav_backbone","--reduction", "8", "--num_enc_layers", "3","--num_dec_layers", "3","--kernel_dim", "3","--emb_dim", "256","--num_objects", "3","--num_workers", "8","--use_query_pos_emb","--use_objectness",  "--use_appearance","--batch_size", "1",  "--pre_norm", "--d_t","3", "--s_t","0.012", "--i_thr","0.35"]


