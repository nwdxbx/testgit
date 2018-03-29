import os
import cv2
import numpy as np
import shutil

srcimg = '/media/f/src_data/Face/ssd-face/srcJPEGImages'
srctxt = '/media/f/src_data/Face/ssd-face/tmp2_txt'
dstimg = '/media/f/src_data/Face/ssd-face/tmpImg'

filetxts = os.listdir(srctxt)
for filtxt in filetxts:
    filimg = filtxt[:-3] + 'jpg'
    image = cv2.imread(os.path.join(srcimg,filimg))
    f = open(os.path.join(srctxt,filtxt),'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        arrayline = line.strip().split()
        x = int(arrayline[0])
        y = int(arrayline[1])
        w = int(arrayline[2])
        h = int(arrayline[3])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)        
        cv2.imwrite(os.path.join(dstimg,filimg),image)
        # if h>50:
        #     x1 = x
        #     y1 = y + (h-w)/4
        #     w1 = w
        #     h1 = h-3*(h-w)/8
            # cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)        
            # cv2.imwrite(os.path.join(dstimg,filimg),image)
            # fr = open(os.path.join(dsttxt,filtxt),'w')
            # result = str(x1)+' '+str(y1)+' '+str(w1)+' '+str(h1)+'\n'
            # fr.writelines(result)
            # fr.close()

    
print 'finish...'