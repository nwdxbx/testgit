#ifndef __GET_STRUCT_FEATURE_HPP__
#define __GET_STRUCT_FEATURE_HPP__
#include "../../include/tuzhenalginterface.hpp"
#include "humanAttribute.hpp"
//#include "caffe/caffe.hpp"


//Input zhfyuan
typedef struct tagTGetFeature
{
    int flag; //0:行人特征; 1：人脸特征; 2：行人特征+行人属性; 3：人脸特征+人脸属性
    //point to featuremap

    //person roi relative to the origin whole image
    Rect roi;
    //small img
    Mat img;
    //keypoints relative pos
    vector<Vec3f> pts;

}TGetFeature;

//int getstructfeatureopen(TGetFeature **ppstGetFeature, int flag);
//int getstructfeatureprocess(TGetFeature *pstGetFeature, PersonInfo &stPersonInfo);
//void getstructfeatureclose(TGetFeature *pstGetFeature);

void attributeAnalysis(cv::Mat &img,std::vector<cv::Vec3f> &pts,cv::Rect &box,PersonInfo &perInfo);



#endif
