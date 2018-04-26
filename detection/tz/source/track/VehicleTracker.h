//
// Created by xulishuang on 18-1-22.
//

#ifndef PROJECT_VEHICLETRACKER_H
#define PROJECT_VEHICLETRACKER_H

#include "tracker.hpp"

class VehicleTracker :public object_tracker{

public:
    VehicleTracker(){};
    virtual void process(cv::Mat srcImg, std::vector<det_input> &objrects, std::vector<tracker_obj_info> &tracker_output);


    void genBodyMask(cv::Mat src,std::vector< cv::Rect > feat,std::vector<cv::Mat> &vecMask);
    void genHist(cv::Mat src,std::vector<cv::Mat> vecMask,std::vector<cv::Mat> &vecHist);

    void similarityMatrix(std::vector<cv::Mat> vecHist, cv::Mat &SM);
    void distanceMatrix(std::vector< cv::Rect > feat, cv::Mat &DM);
};


#endif //PROJECT_VEHICLETRACKER_H
