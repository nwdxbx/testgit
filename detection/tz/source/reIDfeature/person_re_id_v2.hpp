/************************************************************************************
module name:  person_re_id.h
function: imput person image, output its feature
       or input person images, outout its features

Vesion：V2.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2018/01/18           2.0	     shenruixue          create
**************************************************************************************/
#ifndef PERSON_RE_ID_V2_H
#define PERSON_RE_ID_V2_H
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>
#include <map>
#include "../../include/common.h"
using namespace std;

struct Person_Re_ID_det_res{
    vector<float> person_fea;
    int sleeveLength;
    int pantsLength;
    int upclsColor;
    int downclsColor;
    float sleeveLengthScore;
    float pantsLengthScore;
    float coat_color_score;
    float trouser_color_score;
    Person_Re_ID_det_res():sleeveLength(EM_CLOTHES_LENGHT_UNKNOWN),
        pantsLength(EM_CLOTHES_LENGHT_UNKNOWN),
        upclsColor(EM_PERSON_COLOR_UNKNOWN),
        downclsColor(EM_PERSON_COLOR_UNKNOWN),
        sleeveLengthScore(EM_CLOTHES_LENGHT_UNKNOWN),
        pantsLengthScore(EM_CLOTHES_LENGHT_UNKNOWN),
        coat_color_score(EM_PERSON_COLOR_UNKNOWN),
        trouser_color_score(EM_PERSON_COLOR_UNKNOWN)


    {

    }
//    Person_Re_ID_det_res(int count)
//    {
//        person_fea.resize(count);
//    }
};


class Person_Re_ID_v2
{
public:
    static Person_Re_ID_v2 &ins();

    /**
     * @brief process: input an image, output its' feature vector
     * @param imgin: input image, can be a gray and color image
     * @param result: output feature vector, 256 dim
     */
    void process(cv::Mat &imgin, Person_Re_ID_det_res &result);

    /**
     * @brief process: input an image vector, output its' feature vectors
     * @param imgin: input image vector, can be gray and color images
     * @param result: output feature vector, 256 dim per vector
     */
    void process(std::vector<cv::Mat> &imgin, std::vector<Person_Re_ID_det_res> &result);
    template <typename T>
    std::vector<int> sort_indexes(const vector<T> &v);
protected:
    Person_Re_ID_v2(const std::string &net_file,
                 const std::string &model_file,
                 int gpuid=0);
    Person_Re_ID_v2(const Person_Re_ID_v2 &)=delete;
    void preprocess(cv::Mat &img, cv::Mat &resimg);

    int w_;
    int h_;
private:
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr< caffe::Net<float> > net_;
    std::map<int,int>color_map_;

};

#endif // PERSON_RE_ID_V2_H
