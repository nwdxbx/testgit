import os
import cv2
import sys
import mxnet as mx
from mxnet import ndarray as nd

# image_size = [112,112]
# path_imgrec = '/media/e/FrameWork/insightface/datasets/faces_ms1m_112x112/train.rec'
# path_imgidx = '/media/e/FrameWork/insightface/datasets/faces_ms1m_112x112/train.idx'
#outputDir = '/media/f/ms1m'

path_imgrec = '/media/e/FrameWork/insightface/src/data/tmp/test.rec'
path_imgidx = '/media/e/FrameWork/insightface/src/data/tmp/test.idx'
outputDir = '/media/f/101'



imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx,path_imgrec,'r')
s=imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)

print('header0 label', header.label)
header0 = (int(header.label[0]), int(header.label[1]))

imgidx = range(1,int(header.label[0]))

for idx in xrange(int(header.label[0]), int(header.label[1])):
    s = imgrec.read_idx(idx)
    in_header, _ = mx.recordio.unpack(s)

    outputdir = os.path.join(outputDir,str(in_header.id))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for iddx in xrange(int(in_header.label[0]), int(in_header.label[1])):
        ss = imgrec.read_idx(iddx)
        im_header,img = mx.recordio.unpack_img(ss)
        outputPath = os.path.join(outputdir,"%d.jpg"%iddx)
        cv2.imwrite(outputPath,img)

print('finish...')
    