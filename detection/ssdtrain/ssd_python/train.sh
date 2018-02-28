export PYTHONPATH=./:/home/mx/libraries/caffe-ssd/cmakeBuild/install/python:$PYTHONPATH
/home/mx/libraries/caffe-ssd/cmakeBuild/install/bin/caffe train -solver solver.prototxt -weights use4/model_iter_56356.caffemodel
