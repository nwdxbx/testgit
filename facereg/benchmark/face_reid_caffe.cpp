#include "face_reid_caffe.hpp"
namespace alg
{
face_reid_caffe &face_reid_caffe::ins()
{
    static face_reid_caffe obj;
    return obj;
}

face_reid_caffe::face_reid_caffe()
    : simple_caffe<std::vector<float>>::simple_caffe("/media/d/FaceRecognition/convert_to_caffe/resnet_50.prototxt", "/media/d/FaceRecognition/convert_to_caffe/resnet_50.caffemodel", 0)
{ }

void face_reid_caffe::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    //resimg=(img-cv::Scalar(91.4953, 103.8827, 131.0912));//*0.0078125;
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
    //resimg=img/127.5 - cv::Scalar(1, 1, 1);
}


void face_reid_caffe::get_res(const std::vector<caffe::Blob<float>*> &net_res, BATCH_RES &res)
{
    const float *resk=net_res[0]->cpu_data();
    res.resize(net_res[0]->shape(0));
    for (int i=0; i<net_res[0]->shape(0); i++)
    {
        res[i].clear();
        res[i].insert(res[i].begin(), resk, resk+512);
        resk+=512;
    }
}
}
