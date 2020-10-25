PROJECT_SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-${(%):-%x}})/; pwd)

docker_image="pytorch" # not pytorch/pytorch:lates

#  "docker build"
#  example: dkb python3 intro.py
dkb() {
    echo "running dkb ..."

    CMD="docker run --gpus all --rm -ti --ipc=host \
        -v $PROJECT_SOURCE_DIR:$PROJECT_SOURCE_DIR \
        -w `realpath $PWD` -u $(id -u):$(id -g)\
        -e DISPLAY \
        --net=host \
        $docker_image \
        bash -c \"$*\""

    echo ${CMD}

    eval ${CMD}
}