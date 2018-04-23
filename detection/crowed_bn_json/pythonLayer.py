import caffe
from renliu_dataset import renliu_train, renliu_val
import cv2
import numpy as np


class data_layer(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 1

        if self.phase == caffe.TRAIN :
            print '~~~~~~~~~~~~~~~train'
            
            self.batch_loader = renliu_train(self.batch_size)
        else:
            print '~~~~~~~~~~~~~~~test'
           
            self.batch_loader = renliu_val(self.batch_size)

        #top[0].reshape(self.batch_size, 1, 120, 120)
        #top[1].reshape(self.batch_size, 1, 30, 30)
        
    def reshape(self, bottom, top):
        self.img, self.label= self.batch_loader.batch()
        top[0].reshape(self.img.shape[0], self.img.shape[1],
            self.img.shape[2], self.img.shape[3])
        top[1].reshape(self.label.shape[0], self.label.shape[1],
            self.label.shape[2], self.label.shape[3])

    def forward(self, bottom, top):
        #img, label= self.batch_loader.batch()
        #print(self.img.shape)
        top[0].data[...] = self.img
        top[1].data[...] = self.label
        
        
        #cv2.namedWindow("mx")
        #label=self.label[0, 0, :, :]
        #gt=cv2.resize(label, (label.shape[1]*4, label.shape[0]*4))
        #print(np.max(gt))
        #gt=255*gt/np.max(gt)
        
        #cv2.imshow("mx", gt)
        #cv2.waitKey(100)
        #cv2.imwrite('./aaa.png', gt)
        #cv2.imwrite('./bbb.png', (self.img[0, 0, :, :]+1)*127.5)
       
    def backward(self, top, propagate_down, bottom):
        pass
