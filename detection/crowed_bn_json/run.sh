#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt 2>&1|tee train_12_7.log
export PYTHONPATH=/work/project/caffe-stn/python:$PYTHONPATH
export PYTHONPATH=/work2/project/crowed_count/:$PYTHONPATH
/work/project/caffe-stn/build/tools/caffe train -solver solver.prototxt
#/work/project/caffe-stn/build/tools/caffe train -solver solver.prototxt -weights snapshot/crowd_iter_1800.caffemodel 2>&1|tee train_2_28.log


#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt -snapshot examples/ocr/snapshot/mylstm_iter_60000.solverstate 2>&1|tee train_12_11_2.log
