import os
import sys
sys.path.append("/media/e/FrameWork/caffe/python")
import caffe
import numpy as np

net = caffe.Net('/media/e/FrameWork/insightface/models/model-r50-am-lfw/model.prototxt',caffe.TEST)
net.copy_from('/media/e/FrameWork/insightface/models/model-r50-am-lfw/model.caffemodel')

x = net.params
y = x['conv0'][0].data

print y