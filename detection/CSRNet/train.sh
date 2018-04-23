#export LD_LIBRARY_PATH=/home/zhfyuan/libraries/caffe/build/install/lib:$LD_LIBRARY_PATH
#export PYTHONPATH=/media/d/person_reID/add_python_layer:$PYTHONPATH
export PYTHONPATH=./:/media/d/person_reID/caffe_srx/build/install/python:$PYTHONPATH
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
/media/d/person_reID/caffe_srx/build/install/bin/caffe train -gpu 0 -solver solver.prototxt  2>&1  | tee $LOG
