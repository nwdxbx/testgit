#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ffh/work/project/mtcnn-caffe/24net
set -e
~/caffe/build/tools/caffe train \
	 --solver=./solver.prototxt \
	 --weights=./24net.caffemodel
