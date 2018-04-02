import numpy as np
import numpy.random as npr
import sys
import cv2
size = 12
im_shape = 12
net = str(size)
mean = 128
pts_root = '/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/fun_share/data/celeba_cut/'

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

with open('%s/pts_%s.txt'%(net, size), 'r') as f:
    pts2 = f.readlines()

pts_list = []
print '\n'+'part-'+net
cur_ = 0

pts2 = pts2[0].split('\r')
print 'pts2: ',len(pts2)
pts_keep = npr.choice(len(pts2), size=70000, replace=False)
sum_ = len(pts_keep)
cur_=0
for i in pts_keep:
    line = pts2[i]
    view_bar(cur_, sum_)
    cur_+=1
    words = line.split()
    if len(words) < 1:
        continue
    image_file_name = pts_root + words[0]
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=im_shape or w!=im_shape:
        im = cv2.resize(im,(int(im_shape),int(im_shape)))
    points = [0,0,0,0,0,0,0,0,0,0,0]
    # print words[1],words[3],words[5],words[7],words[9]
    # print float(words[1]),float(words[3]),float(words[5]),float(words[7]),float(words[9])
    # print w
    points[1] = float(words[1])/w
    points[2] = float(words[3])/w
    points[3] = float(words[5])/w
    points[4] = float(words[7])/w
    points[5] = float(words[9])/w
    points[6] = float(words[2])/h
    points[7] = float(words[4])/h
    points[8] = float(words[6])/h
    points[9] = float(words[8])/h
    points[10] = float(words[10])/h
    #print points[1],points[2],points[3],points[4],points[5]
    tmp = im[:,:,2].copy()
    im[:,:,2] = im[:,:,0]
    im[:,:,0] = tmp
    im = (im - 127.5)*0.007843137
    im = np.swapaxes(im, 0, 2)
    h,w,ch = im.shape
    label = -1
    roi   = [-1,-1,-1,-1]
    pts	 = [float(points[1]),float(points[ 2]),
            float(points[3]),float(points[ 4]),
            float(points[5]),float(points[6]),
            float(points[7]),float(points[8]),
            float(points[9]),float(points[10])]
    pts_list.append([im,label,roi,pts])
import cPickle as pickle
fid = open("/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/result/12Pnet/imdb/pts.imdb",'w')
pickle.dump(pts_list, fid)
fid.close()