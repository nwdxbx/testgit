from text_dataset import TextDataSet
import cv2
import numpy as np
import re
import math
import random
import re


class renliu_train(TextDataSet):
  def __init__(self, batch_size):
    p = TextDataSet.default_param
    p['filename'] = 'train-sh.txt'
    p['batch_size'] = batch_size
    p['thread_num']=1
    
    self.kernelsz=15
    self.sigma=4
    self.kernel=self._make_gussian_kernel(self.kernelsz, self.sigma)
    TextDataSet.__init__(self, p)
    
    

  def _parse_line(self, line):
    li = re.split(' |\n', line)

    return li

  def _record_process(self, record):
    #print(record[0])
    img = cv2.imread(record[0], cv2.IMREAD_GRAYSCALE)
    #print "###img.w=",img.shape[1]
    #print "###img.h=",img.shape[0]
    if img is None:
      print(record[0])
    img=img.astype(np.float32)

    #add
    img_width = img.shape[1]
    img_height = img.shape[0]
    #print "#####train img.shape[1]=",img_width
    if img_width > 1024:
      shape_bf = (img.shape[1]/16*16, img.shape[0]/16*16)
      shape = (shape_bf[0]/4, shape_bf[1]/4)
      shape1 = (shape[0]/4, shape[1]/4)
      #print "shape=",shape
      #print "shape1=",shape1
      img = cv2.resize(img, shape)
    else:
      shape = (img.shape[1]/4*4, img.shape[0]/4*4)
      shape1 = (shape[0]/4, shape[1]/4)
      img = cv2.resize(img, shape)
    
    #img = img[np.newaxis, np.newaxis, :, :]
    #print "img=",img
    #print "####img.shape=",img.shape
    
    #print "#####train shape=",shape
    #print "#####train shape1=",shape1
    
    fid=open(record[1])
    xy=[]
    for i in fid.readlines():
        li=re.split(' |\n', i)
        if img_width > 1024:
          xy.append((float(li[0])/4, float(li[1])/4))
        else:
          xy.append((float(li[0]), float(li[1])))
    #print "####xy=",xy
   
    #print "####img.shape[2]=",img.shape[2]
    #print "####img.shape[3]=",img.shape[3]

    gt=self._make_ground_truth(xy, (shape[1], shape[0]), self.kernel)
    
    #split_name = re.split('/', record[0])
    #file_name = split_name[7].strip(".jpg")+".csv"
    #np.savetxt(file_name, gt, delimiter = ',')
    #cv2.imshow(file_name, gt*255)
    #cv2.waitKey(0)
    
    #print "####gt_bf=",gt
    #print "gt_shape1=",gt.shape
    
    #cv2.imshow(record[1], gt*255)
    #cv2.waitKey(0)
    
    #split_name = re.split('/', record[0])
    #file_name = split_name[7].strip(".jpg")+"_org.csv"
    #np.savetxt(file_name, gt, delimiter = ',')
    #cv2.imshow(file_name, gt*255)
    
    
    gt = cv2.resize(gt, shape1)
    #print "gt_shape2=",gt.shape
    #cv2.imshow(record[0], gt*255)
    #cv2.waitKey(0)
    
    
    #split_name = re.split('/', record[0])
    #file_name = split_name[7].strip(".jpg")+"_resize.csv"
    #np.savetxt(file_name, gt, delimiter = ',')
    #cv2.imshow(file_name, gt*255)
    
    
    gt = gt*((shape[1]*shape[0])/(shape1[1]*shape1[0]))
    #cv2.imshow(record[0], gt*255)
    #cv2.waitKey(0)
    #print "####gt_type=",type(gt)
    #print("####gt={0}".format(gt))
    
    #split_name = re.split('/', record[0])
    #file_name = split_name[7].strip(".jpg")+"_x16.csv"
    #np.savetxt(file_name, gt, delimiter = ',')
    #cv2.imshow(file_name, gt*255)
    #cv2.waitKey(0)
    
    
    #gt = gt[np.newaxis, np.newaxis, :, :]

    #print "####gt_shape3=",gt.shape
    #print "####gt_after=",gt 
    
    
    #cv2.imwrite('./aaa.png', 255*gt/np.max(gt))
    #cv2.namedWindow("mx")
    #cv2.imshow("mx", 254*gt/np.max(gt))
    #cv2.waitKey(100)
    
    #return (img, gt)

    return [img, gt]
    
  #def _compose(self, list_single):
  #  return list_single[0]
  def _compose(self, list_single):
    batchsz = len(list_single)    
    #print(list_single)
    
    sz=list_single[0][0].shape
    
    hmin=30
    hmax=int(min(int(sz[0]/4), 80))
    if hmin > hmax:
      raise Exception("too small image")
      
    wmin=30
    wmax=int(min(int(sz[1]/4), 80))
    if wmin > wmax:
      raise Exception("too small image")
     
    h=random.randint(hmin, hmax)*4
    w=random.randint(wmin, wmax)*4

    #train_batch_size_t
    batchsize=8
    img=np.zeros((batchsize, 1, h, w), np.float32)
    label=np.zeros((batchsize, 1, h/4, w/4), np.float32)
    for i in range(batchsize):
      y=int(random.randint(0, sz[0]-h)/4)*4
      x=int(random.randint(0, sz[1]-w)/4)*4
      img[i, 0]=list_single[0][0][y:y+h, x:x+w]
      label[i, 0]=list_single[0][1][int(y/4):int(y/4+h/4), int(x/4):int(x/4+w/4)]
      
      #image=list_single[0][0][y:y+h, x:x+w]
      #gt = list_single[0][1][int(y/4):int(y/4+h/4), int(x/4):int(x/4+w/4)]
      #gt=255*gt/np.max(gt)
      
      #cv2.imshow("img", image/255.0)
      #cv2.imshow("gt", gt)
      #cv2.waitKey(0)
    return (img, label)

  def _make_gussian_kernel(self, kernelsz, sigma):
    halfsz=int(kernelsz/2)
    size=int(kernelsz/2)*2+1
    kernel = np.zeros((size, size), np.float32)
    sumall=0.
    for i in range(size):
      for j in range(size):
        val=math.exp(-((i-halfsz)**2+(j-halfsz)**2)/(2.*sigma*sigma))
        kernel[i][j]=val
        sumall+=val
    kernel/=sumall
    return kernel
    
  def _make_ground_truth(self, xy, shape, kernel):
    gt=np.zeros(shape, np.float32)
    #print(self.kernelsz)
    halfsz=int(kernel.shape[0]/2)
    size=int(kernel.shape[0]/2)*2+1
    #print "$$$$xy=",xy
    #print "$$$$shape=",shape
    #print "$$$$kernel=",kernel
    #print "$$$$halfsz=",halfsz
    #print "$$$$size=",size
    for i in xy:
      for y in range(-halfsz, halfsz+1):
        for x in range(-halfsz, halfsz+1):
          imx=int(x+i[0])
          imy=int(y+i[1])
          if (imy >=0 and imy<shape[0] and imx >=0 and imx<shape[1]):
            gt[imy, imx]+=kernel[y+halfsz, x+halfsz]

    return gt

class renliu_val(TextDataSet):
  def __init__(self, batch_size):
    p = TextDataSet.default_param
    p['filename'] = 'val-sh.txt'
    p['batch_size'] = batch_size
    p['thread_num']=1
    
    self.kernelsz=15
    self.sigma=4
    self.kernel=self._make_gussian_kernel(self.kernelsz, self.sigma)
    TextDataSet.__init__(self, p)
    
    

  def _parse_line(self, line):
    li = re.split(' |\n', line)

    return li

  def _record_process(self, record):
    #print(record[0])
    img = cv2.imread(record[0], cv2.IMREAD_GRAYSCALE)
    if img is None:
      print(record[0])
    img=img.astype(np.float32)

    #add
    img_width = img.shape[1]
    img_height = img.shape[0]
    #print "#####img.shape[1]=",img_width
    if img_width > 1024:
      #print "####dayu 640!!!!!!"
      shape_bf = (img.shape[1]/16*16, img.shape[0]/16*16)
      shape = (shape_bf[0]/4, shape_bf[1]/4)
      shape1 = (shape[0]/4, shape[1]/4)
      img = cv2.resize(img, shape)
    else:
      #print "$$$$$$$xiaoyu 640!!!!!!"
      shape = (img.shape[1]/4*4, img.shape[0]/4*4)
      shape1 = (shape[0]/4, shape[1]/4)
      img = cv2.resize(img, shape)
    #img = img[np.newaxis, np.newaxis, :, :]
    
    #print "#####shape=",shape
    #print "#####shape1=",shape1

    fid=open(record[1])
    xy=[]
    for i in fid.readlines():
        li=re.split(' |\n', i)
        if img_width > 1024:
          xy.append((float(li[0])/4, float(li[1])/4))
        else:
          xy.append((float(li[0]), float(li[1])))
        
    gt=self._make_ground_truth(xy, (shape[1], shape[0]), self.kernel)
    
    gt = cv2.resize(gt, shape1)
    
    gt = gt*((shape[1]*shape[0])/(shape1[1]*shape1[0]))
    #print("{0} ####label={1}".format(record[0], gt))
    #gt = gt[np.newaxis, np.newaxis, :, :]
    
    
    #cv2.imwrite('./aaa.png', 255*gt/np.max(gt))
    #cv2.namedWindow("mx")
    #cv2.imshow("mx", 254*gt/np.max(gt))
    #cv2.waitKey(100)
    
    #return (img, gt)
    return [img, gt]
    
    
  #def _compose(self, list_single):
  #  return list_single[0]
  def _compose(self, list_single):
    batchsz = len(list_single)    
    #print(list_single)
    
    sz=list_single[0][0].shape
    
    
    hmin=30
    hmax=int(min(int(sz[0]/4), 80))
    if hmin > hmax:
      raise Exception("too small image")
    wmin=30
    wmax=int(min(int(sz[1]/4), 80))
    if wmin > wmax:
      raise Exception("too small image")
    h=random.randint(hmin, hmax)*4
    w=random.randint(wmin, wmax)*4
  
    #val_batch_size_t
    batchsize=8
    img=np.zeros((batchsize, 1, h, w), np.float32)
    label=np.zeros((batchsize, 1, h/4, w/4), np.float32)
    for i in range(batchsize):
      y=int(random.randint(0, sz[0]-h)/4)*4
      x=int(random.randint(0, sz[1]-w)/4)*4
      img[i, 0]=list_single[0][0][y:y+h, x:x+w]
      label[i, 0]=list_single[0][1][int(y/4):int(y/4+h/4), int(x/4):int(x/4+w/4)]
      
      #image=list_single[0][0][y:y+h, x:x+w]
      #gt = list_single[0][1][int(y/4):int(y/4+h/4), int(x/4):int(x/4+w/4)]
      #gt=255*gt/np.max(gt)
      
      #cv2.imshow("img", image/255.0)
      #cv2.imshow("gt", gt)
      #cv2.waitKey(0)
    return (img, label)
    
  def _make_gussian_kernel(self, kernelsz, sigma):
    halfsz=int(kernelsz/2)
    size=int(kernelsz/2)*2+1
    kernel = np.zeros((size, size), np.float32)
    sumall=0.
    for i in range(size):
      for j in range(size):
        val=math.exp(-((i-halfsz)**2+(j-halfsz)**2)/(2.*sigma*sigma))
        kernel[i][j]=val
        sumall+=val
    kernel/=sumall
    return kernel
    
  def _make_ground_truth(self, xy, shape, kernel):
    gt=np.zeros(shape, np.float32)
    #print(self.kernelsz)
    halfsz=int(kernel.shape[0]/2)
    size=int(kernel.shape[0]/2)*2+1
    for i in xy:
      for y in range(-halfsz, halfsz+1):
        for x in range(-halfsz, halfsz+1):
          imx=int(x+i[0])
          imy=int(y+i[1])
          if (imy >=0 and imy<shape[0] and imx >=0 and imx<shape[1]):
            gt[imy, imx]+=kernel[y+halfsz, x+halfsz]
          
    return gt
    
if __name__ == "__main__":
  d=renliu()
  while(True):
    #print(d.batch())
    x=d.batch()
    print(x[0].shape)
    print(x[1].shape)    
        
        
        
