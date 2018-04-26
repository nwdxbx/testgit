//
// Created by xulishuang on 18-1-23.
//

#ifndef PROJECT_CROWDESTIMATE_H
#define PROJECT_CROWDESTIMATE_H
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

class CrowdEstimate {

public:
    CrowdEstimate();
    static CrowdEstimate &ins();

    void process(const cv::Mat src,cv::Mat &dst,float &headCount);

private:
    std::string netName_;
    std::string modelName_;

    caffe::shared_ptr<caffe::Net<float> > net_;
};


#endif //PROJECT_CROWDESTIMATE_H
