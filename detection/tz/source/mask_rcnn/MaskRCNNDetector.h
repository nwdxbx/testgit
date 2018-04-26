//
// Created by xulishuang on 17-12-28.
//

#ifndef PROJECT_MASKRCNNDETECTOR_H
#define PROJECT_MASKRCNNDETECTOR_H

#include "api.hpp"
#include <vector>

//typedef enum Obj_Type{
//    BACKGROUND=0,
//    PERSON,
//    BICYCLE,
//    CAR,
//    MOTOCYCLE,
//    BUS,
//    TRAIN,
//    TRUCK,
//    BACKPACK,
//    UMBRELLA,
//    HANDBAG,
//    CELLPHONE,
//    SUITCASE,
//    TIE
//};

using caffe::Frcnn::BBox;
class MaskRCNNDetector {

public:
    static MaskRCNNDetector &ins();
    void predict(const cv::Mat &img_in, std::vector<BBox<float> > &results,std::vector<cv::Mat>&masks,
                 std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps,float prob_thresh);
private:
    MaskRCNNDetector( std::string &net_file,
                      std::string &model_file);
    boost::shared_ptr<API::Detector> detector_;
    //API::Set_Config(default_config_file);
    //API::Detector detector(proto_file,model_file);
};


#endif //PROJECT_MASKRCNNDETECTOR_H
