//
// Created by xulishuang on 18-2-24.
//

#ifndef PROJECT_FACELANDMARKPREDICTION_H
#define PROJECT_FACELANDMARKPREDICTION_H

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include "tuzhenalginterface.hpp"


class FaceLandmarkPrediction {

public:
    FaceLandmarkPrediction();

    static FaceLandmarkPrediction &ins();

    void predict(cv::Mat& face,cv::Rect& rt,PersonInfo &personInfo);
private:
    boost::shared_ptr<caffe::Net<float >> face_landmark_net_;
    int w_;
    int h_;
};


#endif //PROJECT_FACELANDMARKPREDICTION_H
