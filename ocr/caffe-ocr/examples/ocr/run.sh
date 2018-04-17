#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt 2>&1|tee train_12_7.log


/media/d/liujin/chepaiRecognition/caffe-ocr/build/tools/caffe train -solver solver.prototxt -weights snapshot/mylstm_iter_f_3100.caffemodel 2>&1|tee train_4_17.log

#sudo ./build/tools/caffe train -solver examples/ocr/solver.prototxt -snapshot examples/ocr/snapshot/mylstm_iter_60000.solverstate 2>&1|tee train_12_11_2.log
