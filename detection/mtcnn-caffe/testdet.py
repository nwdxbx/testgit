def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    #caffe_img = img.copy()-128
    caffe_img = (img.copy()-127.5)-128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools_matrix.calculateScales(img)
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
    det_rectangles = []
    for i in range(image_num):
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools_matrix.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        det_rectangles.extend(rectangle)
    if len(det_rectangles)==0:
        return det_rectangles

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
    roi_prob = out['conv5-2']
    rectangles = tools_matrix.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    #print 'rectangles_24 : ',len(rectangles)
    return rectangles