/************************************************************************************
module name:  humanAttribute.hpp
function: interfaces of human attribute for c++ to call

Vesion：V1.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2017/11/06           1.0	     xbx                 create
**************************************************************************************/
#ifndef HUMANATTRIBUTE_CAFFE_HPP
#define HUMANATTRIBUTE_CAFFE_HPP
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>

struct det_res{
    int sleeveLength;
    int pantsLength;
    int upclsColor;
    int downclsColor;
    // cv::Scalar uprgb;
    // cv::Scalar downrgb;
    float sleeveLengthScore;
    float pantsLengthScore;
    float coat_color_score;
    float trouser_color_score;
};

class humanAttribute_caffe
{
public:
    static humanAttribute_caffe &ins();
    void process(cv::Mat &imgin,det_res &det);  //行人属性前向网络
protected:
    humanAttribute_caffe();
    humanAttribute_caffe(const humanAttribute_caffe &)=delete;
private:
    const std::string netname1_;    //行人属性deploy文件
    const std::string modelname1_;  //行人模型参数文件
    caffe::shared_ptr<caffe::Net<float> > net1_;
    const std::string netname2_;    //颜色deploy文件
    const std::string modelname2_;  //颜色模型参数文件
    caffe::shared_ptr<caffe::Net<float> > net2_;
};
#endif
