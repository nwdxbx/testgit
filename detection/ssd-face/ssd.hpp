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
    int label;
    float score;
};

class ssd_detect_caffe
{
public:
    static ssd_detect_caffe &ins();
    void process(cv::Mat &imgin,ssd_detres &res);
protected:
    ssd_detect_caffe();
    ssd_detect_caffe(const ssd_detect_caffe &)=delete;
private:
    float thresh_;
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr<caffe::Net<float> > net_;
};
#endif

