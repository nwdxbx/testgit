import os
import cv2
import numpy as np

srctxt = '/media/f/src_data/Face/ssd-face/real_txt'
srcJimg = '/media/f/src_data/Face/ssd-face/JPEGImages'

filetxts = os.listdir(srctxt)
for filtxt in filetxts:
    filimg = filtxt[:-3] + 'jpg'
    image = cv2.imread(os.path.join(srcJimg,filimg))
    f = open(os.path.join(srctxt,filtxt),'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        arrayline = line.strip().split()
        cv2.rectangle(image,(int(arrayline[0]),int(arrayline[1])),(int(arrayline[0])+int(arrayline[2]),int(arrayline[1])+int(arrayline[3])),(0,0,255),2)
        print 1.0*int(arrayline[3])/int(arrayline[2])
        cv2.imshow('image',image)
        cv2.waitKey(0)
    
print 'finish...'