/************************************************************************************
module name:  track.hpp
function: imput video stream, output the struct of tracker_obj_info

modify record:
date                 version     author              state
2017/10/25           1.0	     hkj             create
**************************************************************************************/
#ifndef __TRACK_HPP__
#define __TRACK_HPP__
#include "../../include/tuzhenalginterface.hpp"

struct tracker_obj_info
{
    //object pos
    cv::Rect rect;   
    std::vector<Vec3f> points;
    //max score
    float fmaxscore;
    //current score
    float fcurscore;
    size_t objectID;
    bool breport;   

    cv::Mat lastHist;
    //alg module be called times
    size_t trackerlife;   
};

class object_tracker
{
private:
    std::vector<tracker_obj_info> trackers_;
public:
    //static int itvmotion;
    //static int itvstatic;
    object_tracker();
    void process(std::vector< cv::Rect > &feat, std::vector<bool> &shouldAdd);
    void process(cv::Mat srcImg, std::vector< vector<cv::Vec3f> > &points, std::vector<tracker_obj_info> &tracker_output);
private:
    size_t objectIDLast;
    float  fminreportscore;
    void genBodyMask(cv::Mat src,std::vector< cv::Rect > feat,std::vector<cv::Mat> &vecMask);
    void genHist(cv::Mat src,std::vector<cv::Mat> vecMask,std::vector<cv::Mat> &vecHist);

    void similarityMatrix(std::vector<cv::Mat> vecHist, cv::Mat &SM);
    void distanceMatrix(std::vector< cv::Rect > feat, cv::Mat &DM);
    double getScore(std::vector<cv::Vec3f> opts,cv::Rect rect);

};



#endif
