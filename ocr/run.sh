#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt 2>&1|tee train_12_7.log


sudo ../../build/tools/caffe train -solver solver.prototxt -weights snapshot/mylstm_iter_f_4200.caffemodel 2>&1|tee train_1_2.log

#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt -snapshot examples/ocr/snapshot/mylstm_iter_60000.solverstate 2>&1|tee train_12_11_2.log
