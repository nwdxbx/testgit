import os
import sys
sys.path.append('/home/mx/libraries/ocr/ocr/3rdlib/caffe-ocr/python')

import caffe
from caffe.proto import caffe_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2

net = caffe.Net("1.prototxt", caffe.TEST)
net2 = caffe.Net("2.prototxt", caffe.TEST)
net.copy_from('1.caffemodel')

def transform_to_w_and_b(w, b, gamma, beta, mean, var):
    #print(w.shape)
    #print(mean.shape)
    xlen = w.shape[0]
    ylen = w.shape
    zlen = (1,) + ylen[1:]
    zzlen = (xlen, ) + (1,)*len(ylen[1:])
    #print(zzlen)
    w_n = w
    if b is not None:
        b_n = b
    else:
        b_n = np.zeros((xlen, ), np.float32)
    
    if mean is not None and var is not None:
        if gamma is None:
            gamma = np.ones_like(b_n)
        if beta is None:
            beta = np.zeros_like(b_n)
        print('da')
        b_n += beta - gamma / np.sqrt(var+1e-3) * mean
        
        
        gamma = np.tile(gamma.reshape(zzlen), zlen)
        beta = np.tile(beta.reshape(zzlen), zlen)
        mean = np.tile(mean.reshape(zzlen), zlen)
        var = np.tile(var.reshape(zzlen), zlen)
        
        print('gamma: ', gamma.shape)
        print('beta: ', beta.shape)
        print('mean: ', mean.shape)
        print('beta: ', beta.shape)
        
        w_n *= gamma/np.sqrt(var+1e-3)
    return (w_n, b_n)


mapx={
    'conv1_1': ('conv1_1_bn', 'conv1_1_scale'),
    'conv2_1': ('conv2_1_bn', 'conv2_1_scale'),
    'conv3_1': ('conv3_1_bn', 'conv3_1_scale'),
    'conv4_3': ('conv4_3_bn', 'conv4_3_scale'),
    'conv6_2': ('conv6_2_bn', 'conv6_2_scale'),
    'conv4_top_add': ('conv4_top_add_bn', 'conv4_top_add_scale'),
}

for layer_name, layer in net2.params.iteritems():
    for i in range(len(net2.params[layer_name])):
        ##print(net2.params[layer_name][i].data[...])
        try:
            net2.params[layer_name][i].data[...] = net.params[layer_name][i].data
        except Exception as e:
            print(layer_name, i)
        #print(net2.params[layer_name][i].data)


for c, bs in mapx.iteritems():
    print(c, bs)
    w = net.params[c][0].data
    if (len(net.params[c])>1):
        b = net.params[c][1].data
    else:
        b = None
    mw = net.params[bs[0]][0].data/net.params[bs[0]][2].data
    mv = net.params[bs[0]][1].data/net.params[bs[0]][2].data
    gamma = net.params[bs[1]][0].data
    beta = net.params[bs[1]][1].data
    print(c, len(net.params[bs[0]]), len(net.params[bs[1]]))

    w_n, b_n = transform_to_w_and_b(w, b, gamma, beta, mw, mv)
    
    net2.params[c][0].data[...] = w_n #net.params[c][0].data
    #if b is not None:
    #print(b_n)
    net2.params[c][1].data[...] = b_n #net.params[c][1].data
    
net2.save('2.caffemodel')

















