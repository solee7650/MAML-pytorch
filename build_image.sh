# chmod +x build_image.sh

IMAGE="solee/test"

docker build -t $IMAGE:latest .
