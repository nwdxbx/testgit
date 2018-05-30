import os
import sys
import numpy as np
import mxnet as mx
import random 
import cv2
import sklearn

#from mtcnn_detector import MtcnnDetector
#import face_preprocess

prefix = './models/model-r50'
epoch = 271
root_dir = "./data"


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
        model.bind(data_shapes=[('data', (64, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
    
    def get_feature(self,filename):
        img_list = []
        flip_img_list = []
        for i in range(64):
            absname = os.path.join(root_dir,filename)
            img = cv2.imread(absname)         
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(112,112))
            img = np.transpose(img,(2,0,1))
            _img = np.copy(img)
            do_flip(_img)
            img_list.append(img)
            flip_img_list.append(_img)

        embedding = None
        for flipid in [0,1]:
            if flipid == 1:
                data = mx.nd.array(flip_img_list)
            else:
                data = mx.nd.array(img_list)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db,is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy()
            #_embedding = self.model.get_outputs()[0].asnumpy().flatten()
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
        _norm = np.linalg.norm(embedding,axis=1)
        _norm = np.reshape(_norm,(len(_norm),1))
        embedding /=_norm

        return embedding

# if __name__ == "__main__":
#     #img = cv2.imread('./data/Aaron_Eckhart_4.png')
#     featureApp = FaceModel()
#     embedding = featureApp.get_feature('Aaron_Eckhart_4.png')
#     print len(embedding),'\n',embedding

if __name__ == "__main__":
    featureApp = FaceModel()
    files = os.listdir(root_dir)
    for filename in files:
        t1 = cv2.getTickCount()
        feature = featureApp.get_feature(filename)
        t2 = cv2.getTickCount()
        feature_time = (t2-t1)/cv2.getTickFrequency()*1000
        print (feature_time)