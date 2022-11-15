VPATH="/solee"
# UI permisions
xhost +si:localuser:root
CONTAINER="test"
DISPLAY=1
docker run \
            -ti \
            --network host \
            --gpus all \
            --ipc=host \
            -p 8888:8888 \
            -p 6006:6006 \
            --name $CONTAINER \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume="$HOME/.Xauthority:/home/$USER/.Xauthority:rw" \
            --volume="${PWD}:$VPATH" \
            solee/test:latest
#xhost -local:root  # resetting permissions
# --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \