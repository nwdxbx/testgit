import os
import cv2
import numpy as np

src_dir = '/media/f/src_data/Face/ssd-face/tmp'
dst_dir = '/media/f/src_data/Face/ssd-face/tmp_jpeg'
imgfiles = os.listdir(src_dir)

n=0
for imgname in imgfiles:
    image = cv2.imread(os.path.join(src_dir,imgname))
    height,width = image.shape[:2]
    abs_diff = np.abs(2*width-height)
    if 2*width < height:
        dstimage = image[0:2*width,0:width]
    else:
        dstimage = cv2.copyMakeBorder(image,0,abs_diff,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    n=n+1
    cv2.imwrite(os.path.join(dst_dir,imgname),dstimage,[int(cv2.IMWRITE_JPEG_QUALITY),100])
print 'finish......',n