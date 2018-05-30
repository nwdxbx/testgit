# -*- coding: UTF-8 -*-
import sys
import cv2
sys.path.append('/media/e/FrameWork/py-faster-rcnn/caffe-fast-rcnn/python')
import os
import caffe
import numpy as np
import random
import math
import gc
import xml.etree.ElementTree as ET

num_classes = 9
classes = ('__background__', # always index 0
           'kan-dang', 'nei-shi-jing', 'qian-fang', 'yi-biao-pan',
           'you-a', 'you-b', 'zuo-a', 'zuo-b')
classes_to_ind = dict(zip(classes, xrange(num_classes)))

class python_Input_Data(caffe.Layer):
    def setup(self,bottom,top):
        params              = eval(self.param_str_)
        self.num_classes    = params['num_classes']
        self.Jpegs          = params['JPEGImages']
        self.Annos          = params['Annos']
        self.files          = params['listFile']
        self.batchload      = BatchLoader(self.Jpegs,self.Annos,self.files)

        top[0].reshape(1,3,600,1000)
        top[1].reshape(1,3,600,1000)
        top[2].reshape(1,3)
        top[3].reshape(1,5)
        top[4].reshape(1,5)
    
    def reshape(self,bottom,top):
        pass
    
    def forward(self,bottom,top):
        im1,im2,im_info,gt_boxes1,gt_boxes2 = self.batchload.load_next_image()
        height,width        = im1.shape[1:]
        top[0].reshape(1,3,height,width)
        top[1].reshape(1,3,height,width)
        top[2].reshape(im_info.shape[0],im_info.shape[1])
        top[3].reshape(gt_boxes1.shape[0],gt_boxes1.shape[1])
        top[4].reshape(gt_boxes2.shape[0],gt_boxes2.shape[1])

        top[0].data[0,...] = im1
        top[1].data[0,...] = im2
        top[2].data[...] = im_info
        top[3].data[...] = gt_boxes1
        top[4].data[...] = gt_boxes2

    def backward(self,top,propagate_down,bottom):
        pass

def load_pascal_annotation(xmlname):
    config={'use_diff'      : False}
    tree                    = ET.parse(xmlname)
    objs                    = tree.findall('object')
    if not config['use_diff']:
        non_diff_objs = [
            obj for obj in objs if int(obj.find('difficult').text) == 0]
        # if len(non_diff_objs) != len(objs):
        #     print 'Removed {} difficult objects'.format(
        #         len(objs) - len(non_diff_objs))
        objs = non_diff_objs
    num_objs    = len(objs)
    gt_boxes    = np.zeros((num_objs,5),dtype=np.uint16)
    for ix,obj in enumerate(objs):
        bbox    = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        label = classes_to_ind[obj.find('name').text.lower().strip()]
        gt_boxes[ix,:]  = [x1,y1,x2,y2,label]
    return gt_boxes


class BatchLoader(object):
    def __init__(self,JpegsDir,AnnosDir,imglist):
        self.mean           = [127.5,127.5,127.5]
        #self.scale          = 0.0078125
        self.target_size    = 600
        self.max_size       = 1000
        self.JpegsDir       = JpegsDir
        self.AnnosDir       = AnnosDir
        fr_data             = open(imglist,'r')
        imgfiles            = fr_data.readlines()
        fr_data.close()
        self.data_list      = []
        for filename in imgfiles:
            filename1        = filename.strip().split()[0]
            filename2        = filename.strip().split()[1]
            
            self.data_list.append([filename1,filename2])
        random.shuffle(self.data_list)
        self.cur    = 0
        print 'data have been read into memory.\n'

    def load_next_image(self):
        if self.cur == len(self.data_list):
            self.cur = 0 
            random.shuffle(self.data_list)
        filename1        = self.data_list[self.cur][0]      
        endpose1         = filename1.rfind('.')
        xmlname1         = filename1[:endpose1] + '.xml'
        srcImg1          = cv2.imread(os.path.join(self.JpegsDir,filename1))
        srcImg1          = srcImg1.astype(np.float32,copy=True)
        srcImg1          = (srcImg1-self.mean)


        filename2        = self.data_list[self.cur][1]
        endpose2         = filename2.rfind('.')
        xmlname2         = filename2[:endpose2] + '.xml'
        srcImg2          = cv2.imread(os.path.join(self.JpegsDir,filename2))
        srcImg2          = srcImg2.astype(np.float32,copy=True)
        srcImg2          = (srcImg2-self.mean)

        im_shape        = srcImg1.shape
        im_size_min     = np.min(im_shape[0:2])
        im_size_max     = np.max(im_shape[0:2])
        im_scale        = float(self.target_size) / float(im_size_min)
        if np.round(im_scale*im_size_max) > self.max_size:
            im_scale    = float(self.max_size) / float(im_size_max)


        im1              = cv2.resize(srcImg1,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_info         = np.hstack((im1.shape[:2],im_scale))[np.newaxis, :]
        im1              = np.transpose(im1,(2,0,1))
        gt_boxes1        = load_pascal_annotation(os.path.join(self.AnnosDir,xmlname1))
        gt_boxes1[:,:4]  = gt_boxes1[:,:4]*im_scale
        im2              = cv2.resize(srcImg2,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im2              = np.transpose(im2,(2,0,1))
        gt_boxes2        = load_pascal_annotation(os.path.join(self.AnnosDir,xmlname2))
        gt_boxes2[:,:4]  = gt_boxes2[:,:4]*im_scale



        self.cur += 1
        return im1,im2,im_info,gt_boxes1,gt_boxes2