//
// Created by xulishuang on 17-11-24.
//

#ifndef PROJECT_TRACKNET_H
#define PROJECT_TRACKNET_H
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>


class trackNet {

public:
    trackNet();
    static trackNet &ins();
    void process(std::vector< cv::Mat> &imgin, std::vector<std::vector<float>> &result);
    void process(cv::Mat &imgin, std::vector<float> &result);
    void preprocess(cv::Mat &img, cv::Mat &resimg);
    void get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res);
private:
    std::string netName_;
    std::string modelName_;
    int w_;
    int h_;

    caffe::shared_ptr<caffe::Net<float> > net_;
};


#endif //PROJECT_TRACKNET_H
