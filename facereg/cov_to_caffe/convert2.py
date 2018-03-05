import os
import sys
sys.path.append('/media/e/FrameWork/caffe/build/python')
sys.path.append('../code')
filetensor = './model.ckpt-257000'
import caffe
from caffe.proto import caffe_pb2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import face_dataset
from model import face_model

#os.environ['CUDA_VISIBLE_DEVICES']="0"

net = caffe.Net("resnet_50.prototxt", caffe.TEST)

for layer_name, param in net.params.iteritems():
     print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


image_size=[224, 224]
classes=9131
imgs, labels = face_dataset(24, image_size, classes)
y_pred, _ = face_model(imgs, image_size, classes, False)


global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
global_var = filter(lambda x: x.name.find('resnet_v1_50/logits/weights')==-1, global_var)

for var in global_var:
    print(var.name, var.shape)
    

param_map={}
for layer_name, param in net.params.iteritems():
    w = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/weights:0'
    b = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/biases:0'
    gamma = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/BatchNorm/gamma:0'
    beta = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/BatchNorm/beta:0'
    mean = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/BatchNorm/moving_mean:0'
    var = 'resnet_v1_50/'+layer_name.replace('__', '/')+'/BatchNorm/moving_variance:0'
    f = lambda varlist, pattern: filter(lambda v: v.name.find(pattern)!=-1, varlist)
    w_v = f(global_var, w)
    b_v = f(global_var, b)
    gamma_v = f(global_var, gamma)
    beta_v = f(global_var, beta)
    mean_v = f(global_var, mean)
    var_v = f(global_var, var)


    
    param_map[param]=(w_v, b_v, gamma_v, beta_v, mean_v, var_v, layer_name)


for p in param_map:

    ptext='''
        %s:
            w:\t\t\t%s
            b:\t\t\t%s
            gamma:\t\t%s
            beta:\t\t%s
            mean:\t\t%s
            var:\t\t%s
            '''
    def getstr(x):
        if len(x)>0:
            return x[0].name
        else:
            return "None"
    resstr = map(getstr, param_map[p][:-1])
    print( ptext % (p, resstr[0], resstr[1], resstr[2], resstr[3], resstr[4], resstr[5]))





loader = tf.train.Saver(global_var, name='loader')




with tf.Session() as sess:
    tf.global_variables_initializer().run()
    loader.restore(sess, filetensor)

    def extrparam(x):
        if len(x)>0:
            return sess.run(x[0])
        else:
            return None
    def getstr(x):
        if x is not None:
            y=map(lambda x: '%s'%x, x.shape)
            return '(%s)'%(str.join(', ', y))
        else:
            return "None"

    def transform_to_w_and_b(w, b, gamma, beta, mean, var):
        xlen = w.shape[-1]
        ylen = w.shape
        zlen = list(ylen[:-1])
        zlen.append(1)
        w_n=w
        if b is not None:
            b_n = b
        else:
            b_n=np.zeros((xlen,), np.float32)
        
        if mean is not None and var is not None:
            if gamma is None:
                gamma = np.ones_like(b_n)
            if beta is None:
                beta = np.zeros_like(b_n)
                
            b_n += beta - gamma/np.sqrt(var+1e-3)*mean
            #b_n += 1/np.sqrt(var+1e-3)*mean

            gamma = np.tile(gamma, zlen)
            beta = np.tile(beta, zlen)
            mean = np.tile(mean, zlen)
            var = np.tile(var, zlen)
            print('gamma: ', gamma.shape)
            print('beta: ', beta.shape)
            print('mean: ', mean.shape)
            print('var: ', var.shape)

            w_n *= gamma/np.sqrt(var+1e-3)
            #w_n *= 1/np.sqrt(var+1e-3)
            return (w_n, b_n)
        else:
            return (w_n, b_n)

        

    res_map={}
    for p in param_map:
        w, b, gamma, beta, mean, var = map(extrparam, param_map[p][:-1])
        basestr=map(getstr, [w, b, gamma, beta, mean, var])
        print( '%s   ->   %s' % (str.join(' ', basestr), list(p[0].shape)) )

        w_n, b_n = transform_to_w_and_b(w, b, gamma, beta, mean, var)

        if len(w_n.shape)==2:
            w_n = np.transpose(w_n, (1, 0))
        else:
            w_n = np.transpose(w_n, (3, 2, 0, 1))
        print( '%s now ------------------> %s' % ( param_map[p][-1], w_n.shape.__str__() ) )
        res_map[param_map[p][-1]]=(w_n, b_n)


for layer_name, param in res_map.iteritems():
    net.params[layer_name][0].data[...]=param[0]
    #print('before', net.params[layer_name][1].data)
    net.params[layer_name][1].data[...]=param[1]
    #print('now', net.params[layer_name][1].data)

for layer_name, param in net.params.iteritems():
     print param[1].data
net.save('resnet_50.caffemodel')

    




