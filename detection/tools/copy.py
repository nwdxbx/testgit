import os
import cv2
import numpy as np
import shutil

srcimg = '/media/f/src_data/Face/ssd-face/srcJPEGImages'
srctxt = '/media/f/src_data/Face/ssd-face/real_txt'
dstimg = '/media/f/src_data/Face/ssd-face/tmpImg'
dsttxt = '/media/f/src_data/Face/ssd-face/tmp_txt'

filetxts = os.listdir(srctxt)
for filtxt in filetxts:
    filimg = filtxt[:-3] + 'jpg'
    image = cv2.imread(os.path.join(srcimg,filimg))
    f = open(os.path.join(srctxt,filtxt),'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        arrayline = line.strip().split()
        cv2.rectangle(image,(int(arrayline[0]),int(arrayline[1])),(int(arrayline[0])+int(arrayline[2]),int(arrayline[1])+int(arrayline[3])),(0,0,255),2)
        if 1.0*int(arrayline[3])/int(arrayline[2])>1.35:
            shutil.copyfile(os.path.join(srctxt,filtxt),os.path.join(dsttxt,filtxt))
            cv2.imwrite(os.path.join(dstimg,filimg),image)

    
print 'finish...'