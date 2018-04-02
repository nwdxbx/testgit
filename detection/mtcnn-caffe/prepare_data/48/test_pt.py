import cv2
pts_root = '/media/d/fun_share/data/celeba_cut/'
def test_imdata():
    with open('pts_48.txt', 'r') as f:
        pts2 = f.readlines()
    points = [0,0,0,0,0,0,0,0,0,0,0]
    for line in pts2:
        words = line.split()
        image_file_name = pts_root + words[0]
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        points[0] = float(words[1])/w
        points[1] = float(words[3])/w
        points[2] = float(words[5])/w
        points[3] = float(words[7])/w
        points[4] = float(words[9])/w
        points[5] = float(words[2])/h
        points[6] = float(words[4])/h
        points[7] = float(words[6])/h
        points[8] = float(words[8])/h
        points[9] = float(words[10])/h
        pts0= points[0]*w
        pts1= points[5]*h
        pts2= points[1]*w
        pts3= points[6]*h
        pts4= points[2]*w
        pts5= points[7]*h
        pts6= points[3]*w
        pts7= points[8]*h
        pts8= points[4]*w
        pts9= points[9]*h
        cv2.circle(im,(int(pts0),int(pts1)),2,(0,255,0))
        cv2.circle(im,(int(pts2),int(pts3)),2,(0,255,0))
        cv2.circle(im,(int(pts4),int(pts5)),2,(0,255,0))
        cv2.circle(im,(int(pts6),int(pts7)),2,(0,255,0))
        cv2.circle(im,(int(pts8),int(pts9)),2,(0,255,0))
        cv2.imshow('im',im)
        cv2.waitKey(0)

def test_circle():
    anno_file = 'pts_48.txt'
    im_dir = "/home/ffh/work/project/mtcnn-caffe/data/img_align_celeba/"
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    for anno in annotations:
        words = anno.split()
        image_file_name = im_dir + words[0]
        im = cv2.imread(image_file_name)
        for i in range(1,11,2):
            cv2.circle(im,(int(words[i+0]),int(words[i+1])),2,(0,255,0))
        cv2.imshow('im',im)
        cv2.waitKey()

if __name__ == '__main__':
    #test_circle()
    test_imdata()