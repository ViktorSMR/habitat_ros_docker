# syntax=docker/dockerfile:1

###############################################################################
### 1. Builder stage
###############################################################################
ARG CUDA_TAG=12.1.1-cudnn8-devel-ubuntu20.04

FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG MAX_JOBS=4

ARG PYTORCH_CUDA=cu121
ARG PYTORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG NUMPY_VERSION=1.24.4
ARG ME_COMMIT=4b628a7
ARG FAISS_COMMIT=e45ae24

ARG PIP_VERSION=25.0.1
ARG WHEEL_VERSION=0.45.1
ARG SETUPTOOLS_VERSION=69.0.3
ARG NINJA_VERSION=1.11.1.1

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        wget \
        swig \
        ninja-build \
        python3-dev \
        python3-pip \
        libopenblas-dev \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# --Python base---------------------------------------------------------------
RUN python3 -m pip --no-cache-dir install \
        pip==${PIP_VERSION} \
        wheel==${WHEEL_VERSION} \
        setuptools==${SETUPTOOLS_VERSION} \
        ninja==${NINJA_VERSION} \
        numpy==${NUMPY_VERSION}

###############################################################################
### 1a. PyTorch wheel (including torchvision and dependencies)
###############################################################################
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
# MinkowskiEngine require torch installation to build itself
RUN python3 -m pip install --no-cache-dir /wheels/torch*.whl

###############################################################################
### 1b. MinkowskiEngine wheel
###############################################################################
WORKDIR /build/mink
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda-12.1
RUN git clone --recursive https://github.com/alexmelekhin/MinkowskiEngine.git \
        && cd MinkowskiEngine \
        && git checkout 6532dc3 \
        && python3 setup.py bdist_wheel \
                --force_cuda \
                --blas=openblas \
                --dist-dir /wheels

###############################################################################
### 1c. Faiss-GPU wheel
###############################################################################
# upgrade cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5-linux-x86_64.sh && \
    mkdir /opt/cmake-3.26.5 && \
    bash cmake-3.26.5-linux-x86_64.sh --skip-license --prefix=/opt/cmake-3.26.5/ && \
    ln -s /opt/cmake-3.26.5/bin/* /usr/local/bin && \
    rm cmake-3.26.5-linux-x86_64.sh
WORKDIR /build/faiss
RUN git clone https://github.com/facebookresearch/faiss.git \
    && cd faiss \
    && git checkout c3b93749 \
    && cmake -B build . \
        -Wno-dev \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAToolkit_ROOT=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86" \
    && make -C build -j${MAX_JOBS} faiss \
    && make -C build -j${MAX_JOBS} swigfaiss \
    && cd build/faiss/python \
    && python3 setup.py bdist_wheel --dist-dir /wheels

###############################################################################
### 2. Dev/runtime stage
###############################################################################
FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS dev

ENV DEBIAN_FRONTEND=noninteractive

ARG INSTALL_ROS1=false
ENV ROS_DISTRO=noetic

# — lightweight system packages for interactive work —
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python-is-python3 \
        git \
        nano \
        vim \
        sudo \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add -

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.8 cmake=3.14.0

# Setup habitat-sim
RUN /bin/bash -c ". activate habitat; conda install habitat-sim=0.2.3 headless -c aihabitat -c conda-forge"

# Install challenge specific habitat-lab
RUN git clone https://github.com/facebookresearch/habitat-lab -b v0.2.3
RUN /bin/bash -c ". activate habitat; cd habitat-lab; pip install -e habitat-lab/; cd habitat-baselines; pip install habitat-baselines"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/envs/habitat/bin:$PATH  
ENV PYTHONPATH=/opt/conda/envs/habitat/bin/python3 

# Базовые переменные окружения
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/envs/habitat/bin:$PATH  
ENV PYTHONPATH=/opt/conda/envs/habitat/bin/python3 

# Установка базовых зависимостей, ключей, обновление APT и установка пакетов
RUN curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | apt-key add - && \
    apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git python && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl wget less sudo lsof git net-tools nano psmisc xz-utils nemo vim iputils-ping traceroute htop \
        chromium-browser xterm terminator zenity make cmake gcc libc6-dev \
        x11-xkb-utils xauth xfonts-base xkb-data \
        mesa-utils xvfb libgl1-mesa-dri libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
        libglu1 libxv1 \
        libsuitesparse-dev libgtest-dev \
        libeigen3-dev libsdl1.2-dev libarmadillo-dev libsdl-image1.2-dev libsdl-dev \
        software-properties-common supervisor vim-tiny dbus-x11 x11-utils alsa-utils \
        lxde x11vnc gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine firefox libxmu-dev \
        libxext-dev x11proto-gl-dev \
        ninja-build meson autoconf libtool \
        zlib1g-dev libjpeg-dev ffmpeg xorg-dev python-opengl python3-opengl libsdl2-dev swig \
        libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev libpython2.7-dev && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Настройка репозитория CUDA и установка инструментов компиляции
# COPY cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb /
# RUN apt-key del 7fa2af80 && \
#     dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb && \
#     cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin && \
#     mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
#     dpkg -i cuda-keyring_1.1-1_all.deb && \
#     curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add - && \
#     add-apt-repository ppa:ubuntu-toolchain-r/test && \
#     apt update && apt install -y gcc g++

# Добавление архитектуры i386 и сборка CMake из исходников
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt install -y libprotobuf-dev protobuf-compiler build-essential libssl-dev

# Дополнительные APT-зависимости
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common curl wget supervisor sudo vim-tiny net-tools xz-utils dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri lxde x11vnc xvfb gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine firefox && \
    apt-get autoclean -y && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*


# tini for subreap                                   
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1
RUN apt-get update && apt-get install -y \
        git vim \
        python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /
RUN apt-get update && \
    /bin/bash -c ". activate habitat && pip install numpy ipython jupyterlab prompt-toolkit"

WORKDIR /root

RUN apt-get install -y libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y locales && locale-gen "en_US.UTF-8" && \
    pip install imageio

WORKDIR /
# Conda environment

COPY install_nvidia.sh /app/

RUN apt-get -y upgrade && \
    chmod +x /app/install_nvidia.sh && \
    echo "Hello" && NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cuda-cudart-12-1 cuda-compat-12-1 cuda-visual-tools-12-1 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-1 cuda-tools-12-1 cuda-documentation-12-1 cuda-libraries-dev-12-1 && \
    nvcc -V

# libcublas-dev=10.2.1.243-1

RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install numpy


ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

WORKDIR /


RUN /bin/bash -c 'wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -' && \
    apt install -y libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++-multilib \
    libmirclient-dev libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev bison libx11-xcb-dev liblz4-dev libzstd-dev libdwarf-dev && \
    apt-get update && apt list -a lunarg-vktrace

COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
RUN apt-get update
RUN apt-get upgrade -y

RUN pip install matplotlib && \
    pip install tqdm && \
    pip install tabulate && \
    pip install scikit-image && \
    pip install --no-cache-dir Cython && \
    pip install seaborn && \
    pip install ifcfg && \
    pip install imgaug && \
    pip install pycocotools && \
    pip install easydict && \
    pip install pyquaternion && \
    pip install ipywidgets && \
    pip install wandb && \
    pip install lmdb && \
    pip install transformations && \
    pip install scikit-learn && \
    pip install --upgrade numba && \
    pip install omegaconf && \
    pip install keyboard

# WARNING: This allows sudo without password for all users in the sudo group
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# — create a user with the host's UID and GID —
ARG USER_NAME=docker_prism
ARG HOST_UID=1000
ARG HOST_GID=1000
ENV HOME=/home/${USER_NAME}
RUN groupadd --gid ${HOST_GID} ${USER_NAME} \
    && useradd --uid ${HOST_UID} \
               --gid ${HOST_GID} \
               --create-home \
               --shell /bin/bash \
               ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME}

# — copy the compiled wheels and install them —
COPY --from=builder /wheels /tmp/wheels
RUN rm -rf /tmp/wheels/pillow* \
    && python3 -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Install Open3D
RUN python3 -m pip install open3d

RUN apt-get update \
    && apt-get install -y libopenblas-dev ffmpeg libsm6 libxext6

# Install OpenPlaceRecognition
RUN python3 -m pip install --upgrade setuptools \
    && python3 -m pip install --upgrade pip
RUN cd ${HOME} \
    && git clone --branch feat/toposlam https://github.com/OPR-Project/OpenPlaceRecognition \
    && cd OpenPlaceRecognition \
    && python3 -m pip install -e .

# - optional ROS1 Noetic installation -
RUN if [ "$INSTALL_ROS1" = "true" ]; then \
    apt-get update \
    && apt-get install -y lsb-release \
    && apt-get clean all; \
    fi
RUN if [ "$INSTALL_ROS1" = "true" ]; then \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
    && apt -y update \
    && apt install -y ros-${ROS_DISTRO}-desktop-full \
    && apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential; \
    fi

RUN pip install empy==3.3.4 && \
    pip install catkin_pkg && \
    pip install protobuf==3.20.0 && \
    pip install rosnumpy && \
    pip install loguru && \
    pip install memory_profiler

RUN apt-get update && apt-get upgrade -y && apt-get install kmod -y

COPY rgbd4.yaml /habitat-lab/habitat-lab/habitat/config/habitat/simulator/agents/
COPY minkloc3d_nclt.pth /OpenPlaceRecognition/weights/place_recognition/

RUN pip install netifaces && \
    pip install pycryptodomex && \
    pip install git+https://github.com/ros/genpy.git && \
    pip install python-gnupg

RUN mkdir /catkin_ws_initial && mkdir -p /catkin_ws_initial/src

COPY pose_noiser.zip /catkin_ws_initial/src/
COPY pose_to_odom.zip /catkin_ws_initial/src/
COPY depth_image_proc.zip /catkin_ws_initial/src/
COPY toposlam_msgs.zip /catkin_ws_initial/src/
RUN unzip /catkin_ws_initial/src/pose_noiser.zip -d /catkin_ws_initial/src/ && \
    rm -rf /catkin_ws_initial/src/pose_noiser.zip && \
    unzip /catkin_ws_initial/src/pose_to_odom.zip -d /catkin_ws_initial/src/ && \
    rm -rf /catkin_ws_initial/src/pose_to_odom.zip && \
    unzip /catkin_ws_initial/src/toposlam_msgs.zip -d /catkin_ws_initial/src/ && \
    rm -rf /catkin_ws_initial/src/toposlam_msgs.zip && \
    unzip /catkin_ws_initial/src/depth_image_proc.zip -d /catkin_ws_initial/src/ && \
    rm -rf /catkin_ws_initial/src/depth_image_proc.zip

# Adding random for ignoring cache and forcing repos update
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/ViktorSMR/habitat_ros.git -b toposlam_experiments /catkin_ws_initial/src/habitat_ros && \
    git clone https://github.com/KirillMouraviev/PRISM-TopoMap.git /catkin_ws_initial/src/PRISM-TopoMap

RUN mkdir /data_initial && mkdir -p /data_initial/scene_datasets && mkdir -p /data_initial/models

COPY configs /data_initial/configs
COPY datasets /data_initial/datasets
COPY gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth /data_initial/models/

EXPOSE 8888

EXPOSE 6006

COPY image /
#COPY habitat-challenge-data /data_config
ENV HOME /
ENV SHELL /bin/bash

ENV JUPYTER_PASSWORD "jupyter"
ENV JUPYTER_TOKEN "jupyter"

RUN chmod 777 /startup.sh
RUN chmod 777 /usr/local/bin/jupyter.sh
RUN chmod 777 /usr/local/bin/xvfb.sh

ENTRYPOINT ["/startup.sh"]
