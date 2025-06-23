ARG UBUNTU_VERSION=20.04

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgtk2.0-dev \
        libusb-1.0 \
        libgl1-mesa-glx \
        libxrender1 \
        libxrandr2 \
        libxi6 \
        libxinerama1 \
        ocl-icd-libopencl1 \
        clinfo \
        libxcursor1 \
        libxext6 \
        libxfixes3 \
        libvulkan1 \
        ffmpeg \
        gnupg \
        software-properties-common \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install SDK
ARG UBUNTU_VERSION
COPY camera_scripts/99-k4a.rules /etc/udev/rules.d/
COPY camera_scripts/install_azure_kinect_camera_u_${UBUNTU_VERSION}.sh /opt/install.sh
RUN bash /opt/install.sh

# Install Python libraries
RUN apt-get install -y --no-install-recommends python3-opencv python3-setuptools python3-dev
RUN pip3 install --no-cache-dir -U pyk4a typing-extensions numpy

COPY config.py /opt/config.py
COPY camera.py /opt/camera.py
RUN chmod +x /opt/camera.py

ENTRYPOINT ["sh", "-c", "clinfo && python3 /opt/camera.py"]
