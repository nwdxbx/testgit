import numpy as np
import numpy.random as npr
size = 24
net = str(size)
with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos2 = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg2 = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part2 = f.readlines()
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
    
import sys
import cv2
import os
import numpy as np
import gc

cls_list = []
print '\n'+'positive-'+net
cur_ = 0
pos_keep = npr.choice(len(pos2), size=20000, replace=False)
sum_ = len(pos_keep)
for i in pos_keep:
    line = pos2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    tmp = im[:,:,2].copy()
    im[:,:,2] = im[:,:,0]
    im[:,:,0] = tmp
    im = (im - 127.5)*0.007843137
    im = np.swapaxes(im, 0, 2)
    label    = 1
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi])
print '\n'+'negative-'+net
cur_ = 0
neg_keep = npr.choice(len(neg2), size=60000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    tmp = im[:,:,2].copy()
    im[:,:,2] = im[:,:,0]
    im[:,:,0] = tmp
    im = (im - 127.5)*0.007843137
    im = np.swapaxes(im, 0, 2)
    label    = 0
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi]) 
import cPickle as pickle
fid = open("/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/24midimdb_20000/cls.imdb",'w')
pickle.dump(cls_list, fid)
fid.close()
del cls_list
gc.collect()

roi_list = []
print '\n'+'positive-'+net
cur_ = 0
pos_keep = npr.choice(len(pos2), size=20000, replace=False)
sum_ = len(pos_keep)
for i in pos_keep:
    line = pos2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    tmp = im[:,:,2].copy()
    im[:,:,2] = im[:,:,0]
    im[:,:,0] = tmp
    im = (im-127.5)*0.007843137
    im = np.swapaxes(im, 0, 2)
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])

print '\n'+'part-'+net
cur_ = 0
part_keep = npr.choice(len(part2), size=20000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
    line = part2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    tmp = im[:,:,2].copy()
    im[:,:,2] = im[:,:,0]
    im[:,:,0] = tmp
    im = (im-127.5)*0.007843137
    im = np.swapaxes(im, 0, 2)
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])

import cPickle as pickle
fid = open("/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/24midimdb_20000/roi.imdb",'w')
pickle.dump(roi_list, fid)
fid.close()
del roi_list
gc.collect()

# with open('%s/pts_%s.txt'%(net, size), 'r') as f:
#     pts2 = f.readlines()

# pts_list = []
# pts_root = '/home/ffh/work/project/mtcnn-caffe/data/img_align_celeba/'
# print '\n'+'pts-'+net
# cur_ = 0
# pts_keep = npr.choice(len(pts2), size=200000, replace=False)
# sum_ = len(pts_keep)
# cur_=0
# for i in pts_keep:
#     line = pts2[i]
#     view_bar(cur_, sum_)
#     cur_+=1
#     words = line.split()
#     image_file_name = pts_root + words[0]
#     im = cv2.imread(image_file_name)
#     h,w,ch = im.shape
#     if h!=size or w!=size:
#         im = cv2.resize(im,(int(size),int(size)))
#     points[1] = float(words[1])/w
#     points[2] = float(words[3])/w
#     points[3] = float(words[5])/w
#     points[4] = float(words[7])/w
#     points[5] = float(words[9])/w
#     points[6] = float(words[2])/h
#     points[7] = float(words[4])/h
#     points[8] = float(words[6])/h
#     points[9] = float(words[8])/h
#     points[10] = float(words[10])/h
#     tmp = im[:,:,2].copy()
#     im[:,:,2] = im[:,:,0]
#     im[:,:,0] = tmp
#     im = (im - 127.5)*0.007843137
#     im = np.swapaxes(im, 0, 2)
#     h,w,ch = im.shape
#     label = -1
#     roi   = [-1,-1,-1,-1]
#     pts	 = [float(words[1]),float(words[ 2]),
#             float(words[3]),float(words[ 4]),
#             float(words[5]),float(words[6]),
#             float(words[7]),float(words[8]),
#             float(words[9]),float(words[10])]
#     pts_list.append([im,label,roi,pts])
# import cPickle as pickle
# fid = open("/media/ffh/ef830c69-62e9-4b22-8e2a-bc0c289b7e41/mtcnn/data/48/midimdb_20000/pts.imdb",'w')
# pickle.dump(pts_list, fid)
# fid.close()
# del pts_list
# gc.collect()