import sys
import os
import cv2
import caffe
import numpy as np
import random
import torch
import pandas as pd
#from torch.utils.serialization import load_lua
#import py_utils
from multiprocessing import Queue 
from threading import Thread


################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
       
        if self.phase == caffe.TRAIN :
            print '~~~~~~~~~~~~~~~train'
            self.batch_size = 1
            self.img_file = '/home/pengshanzhen/high-quality-densitymap/CSRNet/train_label.txt'
            self.batch_loader = BatchLoader(self.img_file, 'train')
        else:
            print '~~~~~~~~~~~~~~~test'
            self.batch_size = 1
            self.img_file = '/home/pengshanzhen/high-quality-densitymap/CSRNet/val_label.txt'
            self.batch_loader = BatchLoader(self.img_file, 'test')
        #img_root = '/media/pengshanzhen/bb42233c-19d1-4423-b161-e5256766be8e/300/300W_LP'
        #land_mark_root = '/media/pengshanzhen/bb42233c-19d1-4423-b161-e5256766be8e/300/landmarks'
        #self.batch_loader = BatchLoader(img_root, land_mark_root, self.img_w)
        
    def reshape(self, bottom, top):
        self.img, self.den= self.batch_loader.load_next_image()

        top[0].reshape(self.img.shape[0], self.img.shape[1], self.img.shape[2],
                    self.img.shape[3])
        top[1].reshape(self.den.shape[0], self.den.shape[1],
                    self.den.shape[2], self.den.shape[3])
    def forward(self, bottom, top):
        
        top[0].data[ ...] = self.img
        top[1].data[ ...] = self.den
        
        # img=top[0].data[0]
        # img=np.transpose(img, (1,2,0))
        # print img[128,128,0]
        # img=img*128+127.5
        # print img[128,128,0]
        # print(img.shape)
        # cv2.imshow("as", img)
        # cv2.waitKey(1)

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self, img_file, phase):
        self.pts_list = []
        self.phase = phase
        self.mean = 127.5
        with open(img_file)as f:
            img_lines = f.readlines()
        for img_line in img_lines:
            index = img_line.strip().rfind(' ')
            index1 = img_line.strip().rfind('/n')
            img_path = img_line[:index]
            label_path = img_line[index+1:index1]
            #den = pd.read_csv(label_path,sep=',',header=None).as_matrix()
            #den  = den.astype(np.float32, copy=False)
            #count = np.sum(den)

        
            if not os.path.isfile(img_path):
                continue
            #if count == 0:
            #    continue
               
            self.pts_list.append([img_path, label_path])
        random.shuffle(self.pts_list)
        self.pts_cur = 0
        self.record_queue = Queue(maxsize=128)
        self.image_label_queue = Queue(maxsize=128)
        self.thread_num = 6
        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True 
        t_record_producer.start()
        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start() 
        print 'load data done....'
        
    def load_next_image(self): 
        img, den = self.image_label_queue.get()
        return img, den

    def record_producer(self):
        while True:
            if self.pts_cur == len(self.pts_list):
                self.pts_cur = 0
                random.shuffle(self.pts_list)
            self.record_queue.put(self.pts_list[self.pts_cur])
            self.pts_cur += 1

    def record_process(self, cur_data):
        image_file_name = cur_data[0]
        img = cv2.imread(image_file_name,0)
        #img = cv2.imread(image_file_name)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht/8)*8
        wd_1 = (wd/8)*8
        img = cv2.resize(img,(wd_1,ht_1))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))
        #image = cv2.imread(image_file_name)
        
        
        
        label_path = cur_data[1]
        den = pd.read_csv(label_path,sep=',',header=None).as_matrix()
        den  = den.astype(np.float32, copy=False)
        wd_1 = wd_1/8
        ht_1 = ht_1/8
        den = cv2.resize(den,(wd_1,ht_1))                
        den = den * ((wd*ht)/(wd_1*ht_1))
        den = den.reshape((1,1,den.shape[0],den.shape[1]))        
        #label = label.reshape(136)
        return [img, den]

    def record_customer(self):
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)
