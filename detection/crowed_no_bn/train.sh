export PYTHONPATH=./:/home/liujin/caffe/python:$PYTHONPATH
LOG=/media/liujin/bb42233c-19d1-4423-b161-e5256766be8e/new_people/model10/log-`date +%Y-%m-%d-%H-%M-%S`.log 
/home/liujin/caffe/build/tools/caffe train -solver solver.prototxt   2>&1  | tee $LOG $@






