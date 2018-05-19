import os
import sys
import numpy as np
import mxnet as mx
import random 
import cv2
import sklearn

#from mtcnn_detector import MtcnnDetector
#import face_preprocess

prefix = '/media/e/FrameWork/insightface/models/model-r50-am-lfw/model'
epoch = 0


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel(object):
    def __init__(self,prefix=prefix,epoch=epoch):
        image_size = (112,112)
        ctx = mx.gpu(0)
        sym,arg_params,aux_params = mx.model.load_checkpoint(prefix,epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym,context=ctx,label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
    
    def get_feature(self,img):         
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(112,112))
        img = np.transpose(img,(2,0,1))

        embedding = None
        for flipid in [0,1]:
            _img = np.copy(img)
            if flipid == 1:
                do_flip(_img)
            
            input_blob = np.expand_dims(_img,axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db,is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy().flatten()
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
        _norm = np.linalg.norm(embedding)
        embedding /=_norm

        return embedding

if __name__ == "__main__":
    img = cv2.imread('/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/facescrub_aligned/January_Jones/January_Jones_38408.png')
    featureApp = FaceModel()
    embedding = featureApp.get_feature(img)
    print len(embedding),'\n',embedding
