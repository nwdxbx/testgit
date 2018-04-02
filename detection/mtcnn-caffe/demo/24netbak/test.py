import sys
sys.path.append('.')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '24net.prototxt'
caffemodel = '24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '48net.prototxt'
caffemodel = '48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)
caffe.set_device(0)
caffe.set_mode_gpu()

def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    caffe_img = (img_matlab-127.5)*0.007843137
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')
    print '12net: ',len(rectangles)
    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['fc5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    print '24net: ',len(rectangles)
    # if len(rectangles)==0:
    #     return rectangles
    # net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    # crop_number = 0
    # for rectangle in rectangles:
    #     crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
    #     scale_img = cv2.resize(crop_img,(48,48))
    #     scale_img = np.swapaxes(scale_img, 0, 2)
    #     net_48.blobs['data'].data[crop_number] =scale_img 
    #     crop_number += 1
    # out = net_48.forward()
    # cls_prob = out['prob1']
    # roi_prob = out['conv6-2']
    # pts_prob = out['conv6-3']
    # rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    # print 'rects: ',len(rectangles)
    return rectangles

threshold = [0.6,0.6,0.7]
anno_file = 'wider_face_train.txt'
im_dir = "/home/ffh/work/project/mtcnn-caffe/data/WIDER_train/images/"
with open(anno_file, 'r') as f:
    annotations = f.readlines()
#imgpath = "test.jpg"
for annotation in annotations:
    file_name = annotation.strip().split(' ')
    imgpath = im_dir + file_name[0] + '.jpg'
    imgpath = 'test.jpg'
    rectangles = detectFace(imgpath,threshold)
    img = cv2.imread(imgpath)
    draw = img.copy()
    for rectangle in rectangles:
    #print rectangle[0],rectangle[1],rectangle[2],rectangle[3]
        #cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    #for i in range(5,15,2):
    	#cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("test",draw)
    cv2.waitKey()
cv2.imwrite('test_result.jpg',draw)


