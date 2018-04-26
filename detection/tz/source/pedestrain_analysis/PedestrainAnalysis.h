//
// Created by xulishuang on 17-12-28.
//

#ifndef PROJECT_PEDESTRAINDETECTOR_H
#define PROJECT_PEDESTRAINDETECTOR_H
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../../include/tuzhenalginterface.hpp"


typedef struct Pedestrain {
    std::vector<cv::Vec3f> pedestrain_keypoints;   //人体关节点坐标
    Rect pedestrain_roi;      //行人目标框
    float pedestrain_score;
    cv::Mat pedestrain_mask;
};

class PedestrainAnalysis {
public:
    static PedestrainAnalysis &ins();
    void predict(void *pvTuzhenHandle,TuzhenInput &tzInput, TuzhenOutput &tzOutput);
    void predict_v3(void *pvTuzhenHandle,TuzhenInput &tzInput,std::vector<Pedestrain>pedestrains,TuzhenOutput &tzOutput);

    int feat_cal(cv::Mat &img, EFeature_Cal_Type flag, std::string&res);//模型加载放到tuzhen_open同一个线程里边去进行;

//int feat_cal_roi(cv::Mat&img,cv::Rect rt,EFeature_Cal_Type flag, std::string&res);

    int feature_cal_roi(cv::Mat &img, cv::Rect rt,EFeature_Cal_Type flag, std::string&res);//模型加载放到tuzhen_open同一个线程里边去进行;

    int face_feature_attri(cv::Mat &img,PersonInfo& personinfo);//for face only;
private:
    PedestrainAnalysis();

};


#endif //PROJECT_PEDESTRAINDETECTOR_H
