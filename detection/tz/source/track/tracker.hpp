/************************************************************************************
module name:  track.hpp
function: imput video stream, output the struct of tracker_obj_info

modify record:
date                 version     author              state
2017/10/25           1.0	     hkj             create
**************************************************************************************/
#ifndef __TRACK_HPP__
#define __TRACK_HPP__
#include "tuzhenalginterface.hpp"

#include <caffe/caffe.hpp>

struct det_input {
    float x;
    float y;
    float w;
    float h;
    float score;
    int id;
    std::vector<Vec3f> points;
};


struct tracker_obj_info
{
    //rect
    cv::Rect rect;
    //max score
    float fmaxscore;
    //current score
    float fcurscore;
    size_t objectID;
    bool breport;   

    //deep learning for object feature
    std::vector<float> lastOneFeature;
    cv::Mat lastHist;
    //alg module be called times
    size_t trackerlife;

    std::vector<Vec3f> points;
};

class object_tracker
{
protected:
    std::vector<tracker_obj_info> trackers_;
public:
    //static int itvmotion;
    //static int itvstatic;
    object_tracker();
    //static object_tracker &ins();
    //static void loadTrackNet();
    virtual void process(cv::Mat srcImg, std::vector<det_input> &objrects, std::vector<tracker_obj_info> &tracker_output)=0;
protected:
    object_tracker(const std::string &net_file,
                 const std::string &model_file,
                 int gpuid=0);
    object_tracker(const object_tracker &)=delete;
    //void preprocess(cv::Mat &img, cv::Mat &resimg);
    //void get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res);
    int w_;
    int h_;

    //static boost::shared_ptr< caffe::Net<float> > net_;
    size_t objectIDLast;
    float  fminreportscore;

    // deep learning method
    const std::string netname_;
    const std::string modelname_;

    float cosSimilarity(std::vector<float> A,std::vector<float> B);
    void getRectScore(std::vector<det_input> objrects,std::vector<double> &rectScore);
    float similarityTwoFeat(cv::Rect &r1, cv::Rect &r2);

};



#endif
