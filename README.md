# habitat_ros_docker

## Prerequisites

- [Docker](https://www.docker.com/) and [NVIDIA Docker (nvidia-docker)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed.
- An NVIDIA GPU with appropriate drivers is required for accelerated simulation.

## Getting Started

### 1. Build the Docker Image

From the repository root, run:

```bash
sudo ./build.sh
```

### 2. Run the Docker Container

Start the container with ports 5900 and 8900 exposed (VNC and Jupyter respectively):

```bash
sudo ./run.sh 5900 8900
```

By default, the container is started with the following volumes:

```
-v ${HOME}/data/:/data \
-v ${HOME}/catkin_ws/:/catkin_ws \
```
This means that:

* Your local directory ```${HOME}/data/``` will be mounted to ```/data/``` in the container.
* Your local directory ```${HOME}/catkin_ws/``` will be mounted to ```/catkin_ws/``` in the container.

After the container starts, a Jupyter Notebook server will be available at http://localhost:8900.

### 3. Data Preparation
Place topological maps created by [PRISM-TopoMap](https://github.com/KirillMouraviev/PRISM-TopoMap) (in ```mp3d_{scene_name}_rlnav``` format) into the container directory ```/data/graphs/```
Place the required scene files into the container directory: ```/data/scene_datasets/mp3d_toposlam_validation_scenes/```

### 4. Launch ROS and Your Experiment

Inside the container:

Start ROS master:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

In a new terminal, start your experiment:

```bash
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
roslaunch habitat_ros toposlam_experiment_mp3d_4x90.launch scene_name:={scene_name}
```

Replace ```{scene_name}``` with the desired scene you have placed in ```/data/scene_datasets/mp3d_toposlam_validation_scenes/```.

