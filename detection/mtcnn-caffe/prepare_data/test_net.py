import sys
sys.path.append('../demo/')
sys.path.append('.')
import tools_matrix
import caffe
import cv2
import numpy as np
import os
from utils import *

deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
caffe.set_device(0)
caffe.set_mode_gpu()
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
deploy = '24net.prototxt'
caffemodel = '24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/127.5
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools_matrix.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img	
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools_matrix.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools_matrix.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    return rectangles

def show_result():
    threshold = [0.6,0.6,0.7]
    img_path = 'test.jpg'
    rectangles = detectFace(img_path,threshold)
    img = cv2.imread(img_path)
    for rect in rectangles:
        print rect[0],rect[1],rect[2],rect[3]
        cv2.rectangle(img, (int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(255,0,0),1,)
    cv2.imshow("test",img)
    cv2.waitKey()
if __name__ == '__main__':
    show_result()