/************************************************************************************
module name:  person_re_id.h
function: imput person image, output its feature
       or input person images, outout its features

Vesion：V1.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2017/06/01           1.0	     shenruixue          create
**************************************************************************************/
#ifndef PERSON_RE_ID_H
#define PERSON_RE_ID_H
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>

class Person_Re_ID
{
public:
    static Person_Re_ID &ins();

    /**
     * @brief process: input an image, output its' feature vector
     * @param imgin: input image, can be a gray and color image
     * @param result: output feature vector, 256 dim
     */
    void process(cv::Mat &imgin, std::vector<float> &result);

    /**
     * @brief process: input an image vector, output its' feature vectors
     * @param imgin: input image vector, can be gray and color images
     * @param result: output feature vector, 256 dim per vector
     */
    void process(std::vector<cv::Mat> &imgin, std::vector<std::vector<float>> &result);
protected:
    Person_Re_ID(const std::string &net_file,
                 const std::string &model_file,
                 int gpuid=0);
    Person_Re_ID(const Person_Re_ID &)=delete;
    void preprocess(cv::Mat &img, cv::Mat &resimg);
    void get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res);
    int w_;
    int h_;
private:
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr< caffe::Net<float> > net_;

};

#endif // PERSON_RE_ID_H
