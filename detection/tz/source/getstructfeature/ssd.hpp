#ifndef SSDFACEDETECTION_CAFFE_HPP
#define SSDFACEDETECTION_CAFFE_HPP

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>

struct ssd_detres {
    int x1;
    int y1;
    int x2;
    int y2;
    int label;  // 1: front-face  2： back-face
    float score; // confidence of head
};

class ssd_detect_caffe
{
public:
    static ssd_detect_caffe &ins();
    // face detector prob
    // bigger this thresh, then backhead
    void process(cv::Mat &imgin,ssd_detres &res,float thresh = 0.5, float clsthresh = 0.7);
protected:
    ssd_detect_caffe();
    ssd_detect_caffe(const ssd_detect_caffe &)=delete;
    void clsprocess(cv::Mat &facImg,bool &flag);
private:
    float thresh_;
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr<caffe::Net<float> > net_;

    float clsthresh_;
    const std::string clsnetname_;
    const std::string clsmodelname_;
    caffe::shared_ptr<caffe::Net<float> > clsnet_;
};
#endif

