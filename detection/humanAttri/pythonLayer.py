import sys
import cv2
sys.path.append('/media/e/FrameWork/caffe-ssd/python')
import caffe
import numpy as np
import random
import math
import gc

MULTI_NUM = 6



class python_Layer_Data(caffe.Layer):
    def setup(self,bottom,top):
        params                = eval(self.param_str)
        self.batchsize        = params['batch_size']
        self.height           = params['height']
        self.width            = params['width']
        self.data_file        = params['source']
        self.batchload        = BatchLoader(self.data_file,self.height,self.width)
        top[0].reshape(self.batchsize,3,self.height,self.width)
        top[1].reshape(self.batchsize,1)    #pants label
        top[2].reshape(self.batchsize,1)    #sleeve label

    def reshape(self,bottom,top):
        pass

    def forward(self,bottom,top):
        for ii in range(self.batchsize):
            im,pants_label,sleeve_label = self.batchload.load_next_image()
            top[0].data[ii, ...]  = im
            top[1].data[ii, ...]  = pants_label    #pants label
            top[2].data[ii, ...]  = sleeve_label   #sleeve label

    def backward(self,top,propagate_down,bottom):
        pass

class BatchLoader(object):
    def __init__(self, data_file, height, width):
        self.mean          = [127.5, 127.5, 127.5]
        self.height        = height
        self.width         = width
        self.scale         = 0.0078125
        print "reading data into memory....\n"
        fr_data            = open(data_file,'r')
        lines         = fr_data.readlines()
        fr_data.close()
        self.data_list     = []
        for line in lines:
            words          = line.strip().split()
            im             = cv2.imread(words[0])
            im             = cv2.resize(im,(self.width,self.height))
            im             = (im-self.mean)*self.scale
            im             = np.swapaxes(im,0,2)
            im             = np.transpose(im,(0,2,1))
            self.data_list.append([im,int(words[1]),int(words[2])])
        random.shuffle(self.data_list)
        self.cur           = 0
        print "data have been read into memory.\n"
    
    def load_next_image(self):
        if self.cur        == len(self.data_list):
            self.cur       =  0
            random.shuffle(self.data_list)
        im                 =  self.data_list[self.cur][0]
        pants_label             =  self.data_list[self.cur][1]       #pants label
        sleeve_label             =  self.data_list[self.cur][2]       #sleeve  label
        self.cur           += 1
        return im,pants_label,sleeve_label