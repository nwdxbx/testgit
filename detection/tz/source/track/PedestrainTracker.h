//
// Created by xulishuang on 18-1-22.
//

#ifndef PROJECT_PEDESTRAINTRACKER_H
#define PROJECT_PEDESTRAINTRACKER_H

#include "tracker.hpp"

class PedestrainTracker :public object_tracker{

public:
    PedestrainTracker(){};
    virtual void process(cv::Mat srcImg, std::vector<det_input> &persons, std::vector<tracker_obj_info> &tracker_output);

private:
    void distanceMatrix(std::vector< det_input > feat,cv::Mat &DM,double distThresh=300);

    // deep learning method
    //void process2(cv::Mat &imgin, std::vector<float> &result);
    //void process2(std::vector<cv::Mat> &imgin, std::vector<std::vector<float>> &result);
    void similarityMatrix(std::vector<std::vector<float>> objFeature,cv::Mat &SM);


    void genBodyMask(cv::Mat src,std::vector< cv::Rect > feat,std::vector<cv::Mat> &vecMask);
    void genHist(cv::Mat src,std::vector<cv::Mat> vecMask,std::vector<cv::Mat> &vecHist);

    void similarityMatrix(std::vector<cv::Mat> vecHist, cv::Mat &SM);
    void distanceMatrix(std::vector< cv::Rect > feat, cv::Mat &DM);
};


#endif //PROJECT_PEDESTRAINTRACKER_H
