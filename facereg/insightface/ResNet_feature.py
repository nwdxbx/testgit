import sys
import cv2
import numpy as np

sys.path.append('/media/e/FrameWork/caffe/python')
import caffe

deploy_net = '/media/e/FrameWork/insightface/models/caffe-r34-amf/model.prototxt'
caffe_model = '/media/e/FrameWork/insightface/models/caffe-r34-amf/model.caffemodel'

def caffe_process(img):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_net,caffe_model,caffe.TEST)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(112,112),interpolation=cv2.INTER_LINEAR)
    img = img/127.5 -1
    img = np.transpose(img,(2,0,1))
    net.blobs['data'].data[0,...] = img

    out = net.forward()

    feature = net.blobs['feat'].data[0]

    return feature

def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class Resnet_caffe(object):
    def __init__(self,deploy_net=deploy_net,caffe_model=caffe_model):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.net = caffe.Net(deploy_net,caffe_model,caffe.TEST)
    
    def get_feature(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(112,112))
        img = (img-127.5)*0.0078125
        #img = img/127.5 -1
        img = np.transpose(img,(2,0,1))

        embedding = None
        for flipid in [0,1]:
            _img = np.copy(img)
            if flipid==1:
                do_flip(_img)
            #self.net.blobs['data'].data[0,...] = _img  #np.expand_dims(_img,axis=0)
            self.net.blobs['data'].data[...] = np.expand_dims(_img,axis=0)
            out = self.net.forward()
            _embedding = self.net.blobs['fc1'].data[0].flatten()
            if embedding is None:
                embedding = _embedding
            else:
                embedding +=_embedding
        _norm = np.linalg.norm(embedding)
        embedding /= _norm

        return embedding

if __name__ == "__main__":
    img = cv2.imread('/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/facescrub_aligned/January_Jones/January_Jones_38408.png')
    #featureApp = Resnet_caffe(deploy_net,caffe_model)
    featureApp = Resnet_caffe()
    feature = featureApp.get_feature(img)
    print len(feature),'\n',feature