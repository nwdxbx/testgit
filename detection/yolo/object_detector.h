#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include "darknet.h"
#include <iostream>
#include <opencv2/opencv.hpp>

struct det_class{
    float x;
    float y;
    float w;
    float h;
    float score;
    int   id;
};

class yoloDetector
{
public:
    yoloDetector(char* cfgfile,char* weightfile,float num=0.4,float thresh=0.24,float hier_thresh=0.5);
    ~yoloDetector();
    void detector(cv::Mat &img,std::vector<det_class> &det);
    void process(cv::Mat &img,std::vector< std::pair<cv::Rect,float> > &out);
private:
    int max_idx(std::vector<float>&f);
    float iou(det_class &det1, det_class &det2);
    void non_maxima_suppression(std::vector<det_class> &det, float fth);

private:
    float _nms;
    float _thresh;
    float _hier_thresh;
    char* _cfgfile;
    char* _weightfile;
    layer _l;
    network _net;
};


#endif // OBJECT_DETECTOR_H
