/************************************************************************************
module name:  face_interface.h
function: input img of single person, output rect of head and face attributes

Vesion：V1.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2017/11/04           1.0	     lxd                 create
**************************************************************************************/

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "tuzhenalginterface.hpp"
#include "ssd.hpp"


#ifndef FACEFIELDS_FORWARD_FACE_INTERFACE_H
#define FACEFIELDS_FORWARD_FACE_INTERFACE_H
#ifdef GPU
#undef GPU
#undef CUDNN
#endif

class face_interface {
private:
    caffe::shared_ptr<caffe::Net<float>> net_;

    caffe::shared_ptr<caffe::Net<float >> gender_net_;
    caffe::shared_ptr<caffe::Net<float >> age_net_;

    int w_;
    int h_;
    int gpu_id_;

public:
    face_interface();

    void setGPUID(int gpu_id);

    /**
     * initialize a static instance of class face_interface
     * @return a static instance
     */
    static face_interface &ins();

    /**
     * interface for outside functions to call
     *
     * this function is to predict 7 attributes of face and bags
     * all results predicted are decorated in the struct of PersonInfo
     * @param input_img image of person tracked
     * @param personInfo results of caffe and yolov2 predicted
     * @param bag_thres(optional) score threshold to detect bags, 0.4 default
     * @param head_thres(optional) score threshold to detect head, 0.5 default
     */
//    void get_head_infos(cv::Mat &input_img, PersonInfo &personInfo, float attr_thres = 0.7, float bag_thres = 0.4,
//                        float head_thres = 0.5);

    void get_head_infos(cv::Mat &input_img, PersonInfo &personInfo, float attr_thres = 0.7,
                            float head_thres = 0.6);

    void predict_facefields(std::vector<cv::Mat> &input_head_img, std::vector<cv::Mat> &input_body_imgs,
                            vector<PersonInfo> &personInfos,
                            caffe::shared_ptr<caffe::Net<float>> cur_net_, std::string net_tag,
                            float attr_thres = 0.5, EGender_Info gender_info = EM_GENDER_UNKNOWN);

    void predict_facefields(cv::Mat &input_head_img, cv::Mat &input_body_img, PersonInfo &personInfo,
                            float attr_thres = 0.5);
};


#endif //FACEFIELDS_FORWARD_FACE_INTERFACE_H
