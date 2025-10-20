xhost +local:docker || true
xhost +SI:localuser:root
docker run --runtime=nvidia --gpus all -it --rm --name habitat_ros_docker \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--device /dev/nvidia0:/dev/nvidia0 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-env="XAUTHORITY=$XAUTH" \
--volume="$XAUTH:$XAUTH" \
--privileged \
-p 5900:5900 \
-p $2:8888 -e jup_port=$2 \
-v ${HOME}/habitat-lab/data/scene_datasets/:/data/scene_datasets \
-v ${HOME}/TopoSLAM/toposlam_ws/data/habitat_mipt_bags/mp3d_rlnav/:/data/maps \
-v ${HOME}/.Xauthority:/root/.Xauthority:rw \
habitat_ros_image
