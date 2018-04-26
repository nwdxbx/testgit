//
// Created by xulishuang on 17-12-28.
//

#include "MaskRCNNDetector.h"
MaskRCNNDetector::MaskRCNNDetector( std::string &net_file,
                        std::string &model_file)
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(gpuid);

    //net_.reset(new caffe::Net<float>(net_file, caffe::TEST));
    //net_->CopyTrainedLayersFrom(model_file);
    detector_.reset(new API::Detector(net_file,model_file,0.6));
}

MaskRCNNDetector &MaskRCNNDetector::ins()
{
    std::string default_config_file    = "./models/tuzhen_config.json";
    API::Set_Config(default_config_file);
    std::string proto_file             = "./models/mask_rcnn_test_tuzhen_p";
//    std::string model_file             = "./models/vgg16_mask_rcnn_tuzhen_c";
    //std::string proto_file             = "./models/mask_rcnn_test_tuzhen_keypoints_p";
    std::string model_file             = "./models/vgg16_mask_rcnn_tuzhen_keypoints_c";

    static thread_local MaskRCNNDetector obj(proto_file, model_file);
    return obj;
}

void MaskRCNNDetector::predict(const cv::Mat &img_in, std::vector<BBox<float> > &results,std::vector<cv::Mat>&masks,
                               std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps,float prob_thresh)
{
    detector_->predict(img_in,results,masks,keypoints,detect_kps,prob_thresh);
}