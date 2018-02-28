# -*- coding: UTF-8 -*-  
from text_dataset2 import TextDataSet
import cv2
import numpy as np
import re
import math
import random
import sys
import matplotlib.pyplot as plt
import xml.dom.minidom

class ssd_dataset(TextDataSet):
    def __init__(self, batch_size):
        p = TextDataSet.default_param
        p['filename'] = 'trainval.txt'
        p['batch_size'] = batch_size
        p['thread_num']=16
        #p['one_record_to_multisamples']=True
        
        self.imgsize=(160, 96) # w, h
        
        
        TextDataSet.__init__(self, p)

    def _parse_line(self, line):
        li = re.split(' |\n', line)       
        
        return li

    def _record_process(self, record):
        img = cv2.imread(record[0])
        
        imgsz = [img.shape[1], img.shape[0]]
        img = cv2.resize(img, self.imgsize) / 127.5 - 1
        img = img.transpose([2, 0, 1])
        
        obj = self._parsexml(record[1], imgsz)
        #if (obj.shape[0]>1):
        #    print(record)
        
        
        
        #for o in obj:
        #    img = cv2.rectangle(img, (int(o[0]*imgsz[0]), int(o[1]*imgsz[1])), (int(o[2]*imgsz[0]), int(o[3]*imgsz[1])), (255, 0, 0))
        #cv2.imshow('mx', img)
        #cv2.waitKey(0)
        if(obj.shape[0]==0):
            return None
        else:
            return [img, obj]

    def _compose(self, list_single):
        imgs = np.zeros((self.batch_size, 3, self.imgsize[1], self.imgsize[0]), np.float32)
        sz = 0
        for i in range(self.batch_size):
            sz = sz + list_single[i][1].shape[0]
        
        
        labels = np.zeros((1, 1, sz, 8), np.float32)
        
        id = 0
        for i in range(self.batch_size):
            imgs[i] = list_single[i][0]
            obj = list_single[i][1]
            for oid, o in enumerate(obj):
                labels[0, 0, id]=[i, 1, oid, o[0], o[1], o[2], o[3], 0]
                id = id+1
            
            
        
        
        #for i in range(self.batch_size):
        #    imgs[i] = list_single[i]

        return [imgs, labels]
    def _parsexml(self, filename, imgsz):
        dom = xml.dom.minidom.parse(filename)
        root = dom.documentElement
        res = root.getElementsByTagName('object')
        xy = np.zeros((len(res), 4), np.float32)
        for id, r in enumerate(res):
            x1 = float(r.getElementsByTagName('xmin')[0].firstChild.data)
            y1 = float(r.getElementsByTagName('ymin')[0].firstChild.data)
            x2 = float(r.getElementsByTagName('xmax')[0].firstChild.data)
            y2 = float(r.getElementsByTagName('ymax')[0].firstChild.data)
            xy[id]=[x1/imgsz[0], y1/imgsz[1], x2/imgsz[0], y2/imgsz[1]]
            
        return xy
            

    
if __name__ == "__main__":
    d=ssd_dataset(4)
    while(True):
        res = d.batch()
        print(res[0].shape)
        print(res[1].shape)

        
        
        
        
        
        
        
        
