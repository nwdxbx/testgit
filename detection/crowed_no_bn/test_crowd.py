import cv2
import numpy as np
import caffe
import os
import pandas as pd



caffe.set_mode_gpu()

mae = 0.0
mse = 0.0
output_dir = '/home/pengshanzhen/test_new_people/18/'
gt_count_list =[]
et_count_list =[]
data_path = '/home/pengshanzhen/test_new_people/data/test_data/images/'
label_path = '/home/pengshanzhen/test_new_people/data/test_data/ground_truth_csv/'
filelist = os.listdir(data_path)

attrNet = caffe.Net("/home/pengshanzhen/test_new_people/deploy_nobn.prototxt", "/home/pengshanzhen/test_new_people/model_iter_142500.caffemodel", caffe.TEST)
for fname in filelist:
    
    img_path =os.path.join(data_path,fname)
    img = cv2.imread(img_path,0)
    img = img.astype(np.float32, copy=False)

    den = pd.read_csv(os.path.join(label_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()          
    den = den.astype(np.float32, copy=False)
    
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht/4)*4
    wd_1 = (wd/4)*4
    img = cv2.resize(img,(wd_1,ht_1))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = np.transpose(img,(2,0,1))
    wd_2 = wd_1/4
    ht_2 = ht_1/4
    den = cv2.resize(den,(wd_2,ht_2))                
    den = den * ((wd*ht)/(wd_2*ht_2))
    gt_data = den.reshape((1,1,den.shape[0],den.shape[1]))  
    attrNet.blobs['data'].reshape(1, 1,ht_1,wd_1)
    attrNet.blobs['data'].data[0][0] = img
    print fname,ht_1,wd_1
    out = attrNet.forward()
    #print 'forward'
    pre_density_map = out['fuse_conv']
    
    gt_count = np.sum(gt_data)
    
    gt_count_list.append(gt_count)
    et_count = np.sum(pre_density_map)

    
    
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    
    pre_density_map = 255*pre_density_map/np.max(pre_density_map)
    
    pre_density_map= pre_density_map[0][0]
  
    cv2.imwrite(os.path.join(output_dir,'output_'+fname.split('.')[0]+'.png'),pre_density_map)

num_samples = len(filelist)
print(num_samples)
mae = mae/num_samples
mse = np.sqrt(mse/num_samples)
print(gt_count_list)
print(et_count_list)
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open('/home/pengshanzhen/test_new_people/18/results.txt', 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()
