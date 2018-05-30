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

# num_classes = 21
# classes = ('__background__', # always index 0
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
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
        top[1].reshape(1,3)
        top[2].reshape(1,5)
    
    def reshape(self,bottom,top):
        pass
    
    def forward(self,bottom,top):
        im,im_info,gt_boxes = self.batchload.load_next_image()
        height,width        = im.shape[1:]
        top[0].reshape(1,3,height,width)
        top[1].reshape(im_info.shape[0],im_info.shape[1])
        top[2].reshape(gt_boxes.shape[0],gt_boxes.shape[1])

        top[0].data[0,...] = im
        top[1].data[...] = im_info
        top[2].data[...] = gt_boxes

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
            filename        = filename.strip().split()[0]
            
            self.data_list.append(filename)
        random.shuffle(self.data_list)
        self.cur    = 0
        print 'data have been read into memory.\n'

    def load_next_image(self):
        if self.cur == len(self.data_list):
            self.cur = 0 
            random.shuffle(self.data_list)
        filename        = self.data_list[self.cur]
        endpose         = filename.rfind('.')
        xmlname         = filename[:endpose] + '.xml'
        srcImg          = cv2.imread(os.path.join(self.JpegsDir,filename))
        srcImg          = srcImg.astype(np.float32,copy=True)
        srcImg          = (srcImg-self.mean)

        im_shape        = srcImg.shape
        im_size_min     = np.min(im_shape[0:2])
        im_size_max     = np.max(im_shape[0:2])

        im_scale        = float(self.target_size) / float(im_size_min)
        if np.round(im_scale*im_size_max) > self.max_size:
            im_scale    = float(self.max_size) / float(im_size_max)
        im              = cv2.resize(srcImg,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)

        im_info         = np.hstack((im.shape[:2],im_scale))[np.newaxis, :]
        im              = np.transpose(im,(2,0,1))
        gt_boxes        = load_pascal_annotation(os.path.join(self.AnnosDir,xmlname))
        gt_boxes[:,:4]  = gt_boxes[:,:4]*im_scale
        self.cur += 1
        return im,im_info,gt_boxes