//
// Created by xulishuang on 17-12-11.
//
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "../include/tuzhenfeature.h"
#include "../include/tuzhenalginterface.hpp"

void faceAttributeDemo()
{
    string file="/home/xulishuang/公共的/share/xx14.jpg";
    cv::Mat frame = cv::imread(file);
    PersonInfo personinfo;
    LOG(INFO) << personinfo.coat_color << "\n";
    LOG(INFO) << personinfo.trouser_color << "\n";
    LOG(INFO) << personinfo.pos_type << "\n";
    LOG(INFO) << personinfo.backpack << "\n";
    LOG(INFO) << personinfo.sleeve_length << "\n";
    LOG(INFO) << personinfo.pants_length << "\n";
    LOG(INFO) << personinfo.head_type << "\n";
    LOG(INFO) << personinfo.gender_info << "\n";
    LOG(INFO) << personinfo.age_info << "\n";
    LOG(INFO) << personinfo.age_value << "\n";
    LOG(INFO) << personinfo.glass_info << "\n";
    LOG(INFO) << personinfo.hat_info << "\n";

    face_feature_attri(frame,personinfo);

    LOG(INFO) << personinfo.bgetface;
    LOG(INFO) << personinfo.facefea;
}

int main(void)
{
//    caffe::shared_ptr<caffe::Net<float> > net_;
//    net_.reset(new caffe::Net<float>("./models/humanAttribute.prototxt",caffe::TEST));
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    boost::shared_ptr< caffe::Net<float> > net_ = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>("/work/dev/experiments/py-mask-rcnn/models/coco/ResNet50/mask_rcnn_end2end/train.prototxt", caffe::TEST));
//    std::ifstream t("./models/face_reid.caffemodel");
//    std::stringstream buffer;
//    buffer << t.rdbuf();
//    std::string contents(buffer.str());
//     net_->CopyTrainedLayersFromString(contents);
     //NetParameter param;
     //param.ParseFromString(contents);
//    std::cout << "test mode....";
//    void *pvTuzhenHandle;
//    //get para from configure file
//    TuzhenPara para;
//    para.deviceID = 0;
//    para.intervalFrames = 25;
//    para.eoptfun = EM_OPTIONAL_ITEM_FACE_DETECT | EM_OPTIONAL_ITEM_CROWD_DETECT;
//
//    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
//
//
//    faceAttributeDemo();
//
//    tuzhen_close(pvTuzhenHandle);
//    std::cout << "release mode";
    return 0;
}