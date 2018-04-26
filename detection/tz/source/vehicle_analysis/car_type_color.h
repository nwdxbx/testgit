//
// Created by huanglkaijun on 17-12-26.
//

#ifndef PROJECT_TRACKNET_H
#define PROJECT_TRACKNET_H
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

typedef  enum {
    VEHICLE_BRAND=0,
    VEHICLE_COLOR
}VehicleDetectionType;

class car_type_color {

public:

    static car_type_color &ins();
    void process(cv::Mat &imgin, std::vector<float> &output,VehicleDetectionType type);
    void preprocess(cv::Mat &img, cv::Mat &resimg);

    void car_id(cv::Mat &src,std::vector<std::string> &vecResult,float type_thresh=0.1,float color_thresh=0.1);
    
private:
    car_type_color();
    int w_;
    int h_;

    boost::shared_ptr<caffe::Net<float> > vehicle_brand_net_;
    boost::shared_ptr<caffe::Net<float> > vehicle_color_net_;
    std::map<int,std::string> labelMap;
    std::map<int,std::string> colorMap;
};


#endif //PROJECT_TRACKNET_H
