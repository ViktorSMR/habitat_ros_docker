FROM nvidia/cudagl:11.0-devel-ubuntu20.04

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
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
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
        libglu1 libglu1:i386 libxv1 libxv1:i386 \
        libsuitesparse-dev libgtest-dev \
        libeigen3-dev libsdl1.2-dev libarmadillo-dev libsdl-image1.2-dev libsdl-dev \
        software-properties-common supervisor vim-tiny dbus-x11 x11-utils alsa-utils \
        lxde x11vnc gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine firefox libxmu-dev \
        libssl-dev:i386 libxext-dev x11proto-gl-dev \
        ninja-build meson autoconf libtool \
        zlib1g-dev libjpeg-dev ffmpeg xorg-dev python-opengl python3-opengl libsdl2-dev swig \
        libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev libpython2.7-dev && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Настройка репозитория CUDA и установка инструментов компиляции
COPY cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb /
RUN apt-key del 7fa2af80 && \
    dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin && \
    mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add - && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && apt install -y gcc g++

# Добавление архитектуры i386 и сборка CMake из исходников
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt install -y libprotobuf-dev protobuf-compiler build-essential libssl-dev
    # /bin/bash -c 'wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3.tar.gz && \
    # tar -zxvf cmake-3.21.3.tar.gz && cd cmake-3.21.3 && ./bootstrap && make && sudo make install'

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


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

#RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
#WORKDIR /detectron2_repo
#RUN git reset --hard 9eb4831f742ae6a13b8edb61d07b619392fb6543
WORKDIR /


#RUN pip install -e detectron2_repo

RUN /bin/bash -c 'wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -' && \
    apt install -y libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++-multilib \
    libmirclient-dev libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev bison libx11-xcb-dev liblz4-dev libzstd-dev libdwarf-dev && \
    apt-get update && apt list -a lunarg-vktrace

COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

# RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
#     wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.170-bionic.list http://packages.lunarg.com/vulkan/1.2.170/lunarg-vulkan-1.2.170-bionic.list && \
#     apt update && apt install -y vulkan-sdk && apt upgrade -y && apt autoremove -y   

RUN apt-get update
RUN apt-get upgrade -y


#RUN /bin/bash -c 'git clone --recursive https://github.com/shacklettbp/bps3D; \
#cd bps3D; \
#mkdir build; \
#cd build; \
#cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..; \
#make' 

#add_definitions(-D GLM_ENABLE_EXPERIMENTAL)

#RUN /bin/bash -c 'git clone --recursive https://github.com/shacklettbp/bps-nav.git; \
#cd bps-nav; \
#cd simulator/python/bps_sim; \
#pip install -e . # Build simulator; \
#cd ../bps_pytorch; \
#pip install -e . # Build pytorch integration; \
#cd ../../../; \
#pip install -e .'

RUN apt-get update
# RUN apt-get install -y kmod kbd

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
# RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /
# # services like lxde, xvfb, x11vnc, jupyterlab will be started

RUN git clone https://github.com/naver/debit.git /debit && \
    cd /debit && \
    git clone https://github.com/naver/croco src/croco && \
    find src/croco -type d -exec touch {}/__init__.py \; && \
    find src/croco/models -name "*.py" -exec sed -ie 's/^from models/from /' {} \; && \
    pip install -e . && \
    mkdir -p out/ckpt/hab_bl/imgnav && cd out/ckpt/hab_bl/imgnav && \
    curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_large.pth && \
    curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_base.pth && \
    curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_small.pth && \
    curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_tiny.pth

WORKDIR /

RUN cd habitat-lab/habitat-baselines && \
    pip install .


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
RUN apt-get update
RUN apt-get install ros-noetic-desktop-full -y
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt-get install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
RUN apt-get install python3-empy -y
RUN pip install empy catkin_pkg
RUN pip install rosdep
RUN apt-get update
RUN rosdep init
RUN rosdep update

ARG MAX_JOBS=4

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    sudo \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    swig \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev\
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    locales \
    language-pack-en \
    language-pack-ru \
    && rm -rf /var/lib/apt/lists/*

# Ensure locale environment variables are properly set
RUN locale-gen en_US.UTF-8 ru_RU.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# upgrade pip
ARG PIP_VERSION=23.3.2
ARG SETUPTOOLS_VERSION=69.0.3
RUN pip install pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION}

ARG NINJA_VERSION=1.11.1.1
RUN pip install ninja==${NINJA_VERSION}
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda-12.1
RUN git clone --recursive "https://github.com/alexmelekhin/MinkowskiEngine.git" && \
    cd MinkowskiEngine && \
    git checkout 6532dc3 && \
    python3 setup.py install --force_cuda --blas=openblas && \
    cd .. && \
    rm -rf MinkowskiEngine

RUN pip install cmake==3.26.4


RUN git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    git checkout c3b93749 && \
    cmake -B build . \
        -Wno-dev \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
        -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86" && \
    make -C build -j${MAX_JOBS} faiss && \
    make -C build -j${MAX_JOBS} swigfaiss && \
    cd build/faiss/python && python3 setup.py install && \
    cd / && \
    rm -rf faiss

RUN pip install onnxruntime-gpu==1.18 --extra-index-url \
    https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ && \
    pip install tensorrt==8.6.0 && \
    pip install torch-tensorrt==2.1.0 --extra-index-url https://download.pytorch.org/whl/test/cu121 && \
    pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com && \
    pip install onnx

RUN git clone https://github.com/isl-org/Open3D.git && cd Open3D && git checkout c8856fc
WORKDIR /Open3D
RUN bash util/install_deps_ubuntu.sh assume-yes && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGLIBCXX_USE_CXX11_ABI=OFF \
        -DBUILD_CUDA_MODULE=ON \
        -DBUILD_PYTORCH_OPS=ON \
        -DBUILD_TENSORFLOW_OPS=OFF \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. && \
    make -j${MAX_JOBS} && \
    make install -j${MAX_JOBS}
ARG YAPF_VERSION=0.43.0
RUN pip install yapf==${YAPF_VERSION}
RUN cd build && make install-pip-package -j${MAX_JOBS}
    
# install pytorch3d
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
    
# install PointMamba requirements
WORKDIR /
ARG CAUSALCONV1D_VERSION=1.4.0
ARG MAMBA_VERSION=1.2.2
RUN git clone https://github.com/alexmelekhin/PointMamba.git && \
    cd PointMamba && \
    pip install -r requirements.txt && \
    cd pointmamba/extensions/chamfer_dist && \
    python setup.py install && \
    cd ../emd && \
    python setup.py install && \
    pip install "git+https://github.com/alexmelekhin/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" && \
    pip install causal-conv1d==${CAUSALCONV1D_VERSION} && \
    pip install mamba-ssm==${MAMBA_VERSION} && \
    cd / && \
    rm -rf PointMamba
    
# install PaddlePaddle and PaddleOCR for OCR tasks
RUN pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
ARG PADDLEOCR_VERSION=2.10.0
RUN pip install paddleocr==${PADDLEOCR_VERSION}

RUN git clone https://github.com/OPR-Project/OpenPlaceRecognition.git -b feat/toposlam && \
    cd OpenPlaceRecognition && \
    git submodule update --init && \
    pip install -e .

ARG DISTRO_VERSION=1.9.0
RUN pip install distro==${DISTRO_VERSION}

RUN pip install --user --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# COPY requirements.txt .
# RUN pip install -r requirements.txt && \
#     rm requirements.txt


# RUN pip install -e ~/OpenPlaceRecognition

# RUN mkdir -p /catkin_ws/src

# COPY habitat_ros /catkin_ws/src/habitat_ros
# COPY toposlam_msgs /catkin_ws/src/toposlam_msgs
# COPY PRISM-TopoMap /catkin_ws/src/PRISM-TopoMap

# RUN cd /catkin_ws/src && \
#     cd habitat_ros && \
#     pip install -r requirements.txt 

RUN pip install empy==3.3.4 && \
    pip install protobuf==3.20.0 && \
    pip install rosnumpy && \
    pip install loguru && \
    pip install memory_profiler

# Переходим в корневую папку рабочего пространства и собираем все пакеты с помощью catkin_make
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash" && \
#     cd /catkin_ws && /bin/bash -c "catkin_make"

RUN apt-get update && apt-get upgrade -y && apt-get install kmod -y

# jupyterlab port
EXPOSE 8888
# tensorboard (if any)
EXPOSE 6006
# startup
COPY image /
#COPY habitat-challenge-data /data_config
ENV HOME /
ENV SHELL /bin/bash

# no password and token for jupyter
ENV JUPYTER_PASSWORD "jupyter"
ENV JUPYTER_TOKEN "jupyter"

RUN chmod 777 /startup.sh
RUN chmod 777 /usr/local/bin/jupyter.sh
RUN chmod 777 /usr/local/bin/xvfb.sh


ENTRYPOINT ["/startup.sh"]
