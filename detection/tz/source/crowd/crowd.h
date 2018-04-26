/************************************************************************************
module name:  crowd.h
function: imput video stream, output count of heads and the hot map

modify record:
date                 version     author              state
2017/11/01           1.0	     hkj&wxc             create
**************************************************************************************/


#ifndef CROWD_H
#define CROWD_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>


class crowd
{
public:
    crowd();
    static crowd &ins();

    //伪彩色处理
    void falColor(cv::Mat src,cv::Mat &dst);
    //前向函数
    void process(const cv::Mat src,cv::Mat &dst,float &headCount);

private:
    std::string netName_;
    std::string modelName_;

    caffe::shared_ptr<caffe::Net<float> > net_;
};

#endif // CROWD_H
