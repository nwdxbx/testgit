import os
import cv2
import math
import numpy as np

srcdir   = '/media/f/src_data/Face/realface/JPEGImages'
jpegImgs = os.listdir(srcdir)
backImgs = os.listdir('/media/f/src_data/Face/realface/class-face/back')
# degree90Imgs = os.listdir('/media/f/src_data/Face/1207/degree90')
frontImgs = os.listdir('/media/f/src_data/Face/realface/class-face/front')
sideImgs = os.listdir('/media/f/src_data/Face/realface/class-face/side')

labdir = '/media/f/src_data/Face/1207/labels'
dstdir = '/media/f/src_data/Face/realface/real_txt'
# siddir = '/media/f/src_data/Face/realface/sideface_txt'

def bbox(bbox,size):
    cenx = bbox[0]
    ceny = bbox[1]
    w    = bbox[2]
    h    = bbox[3]
    width = int(w*size[0])
    height = int(h*size[1])
    x = int(cenx*size[0] - width/2)
    y = int(ceny*size[1] - height/2)

    x = min(max(0,x),size[0])
    y = min(max(0,y),size[1])
    width = min(max(0,x+width),size[0]) - x
    height = min(max(0,y+height),size[1]) - y
    return x,y,width,height

# for img in jpegImgs:
#     label = -1
#     if img in frontImgs:
#         label = 0
#     elif img in backImgs:
#         label = 1
#     elif img in sideImgs:
#         label = 2
#     # image = cv2.imread(os.path.join(srcdir,img))
#     txtfile = img[:-3] + 'txt'
#     f = open(os.path.join(labdir,txtfile),'r')
#     lines = f.readlines()
#     f.close() 
#     fw = open(os.path.join(dstdir,txtfile),'w')   
#     for line in lines:
#         arrayline = line.strip().split()
#         if str(arrayline[0]) == '0' or str(arrayline[0]) == '1':
#             x= int(arrayline[1])
#             y= int(arrayline[2])
#             w= int(arrayline[3])
#             h= int(arrayline[4])
#             strline = str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
#             # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#             fw.writelines(strline)
#     # cv2.imshow('image',image)
#     # cv2.waitKey(0)
#     fw.close()

# print ('finish...')



for img in jpegImgs:
    label = -1
    if img in frontImgs:
        label = 0
    elif img in backImgs:
        label = 1
    # elif img in degree90Imgs:
    #     label = 2
    elif img in sideImgs:
        label = 2
    image = cv2.imread(os.path.join(srcdir,img))
    height,width =image.shape[:-1]
    txtfile = img[:-3] + 'txt'
    f = open(os.path.join(labdir,txtfile),'r')
    lines = f.readlines()
    f.close()
    # if label == 3:
    #     fw = open(os.path.join(siddir,txtfile),'w')
    # else:
    #     fw = open(os.path.join(dstdir,txtfile),'w')  
    fw = open(os.path.join(dstdir,txtfile),'w')   
    for line in lines:
        arrayline = line.strip().split()
        if str(arrayline[0]) == '0' or str(arrayline[0]) == '1':
            box = [float(arrayline[1]),float(arrayline[2]),float(arrayline[3]),float(arrayline[4])]
            size = [int(width),int(height)]
            x,y,w,h = bbox(box,size)
            strline = str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            fw.writelines(strline)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    fw.close()

print ('finish...')