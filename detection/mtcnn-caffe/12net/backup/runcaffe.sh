#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ffh/work/project/mtcnn-caffe/12net:/home/ffh/caffe/python
export LD_LIBRARY_PATH=$LD_LIBRARY:/usr/local/cuda-8.0/lib64
set -e
/home/ffh/caffe/build/tools/caffe train --solver=./solver.prototxt --weights 12net-cls-only.caffemodel
