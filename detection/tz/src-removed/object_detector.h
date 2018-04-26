/************************************************************************************
module name:  object_detector.h
function: interfaces of darknet's yolov2 decorated for c++ to call

Vesion：V1.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2017/11/04           1.0	     xbx, lxd            create
**************************************************************************************/
#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <darknet.h>
#include <iostream>
#include <opencv2/opencv.hpp>

struct det_class {
    float x;
    float y;
    float w;
    float h;
    float score;
    int id;
};

class yoloDetector {
public:
    yoloDetector();

    /**
     * initialize yolov2's cfgfile(*.cfg) and weightfile(*.weights)
     * @param cfgfile yolo's configuration file
     * @param weightfile yolo's weighted fiel
     * @param nms
     * @param thresh
     * @param hier_thresh
     */
    yoloDetector(char *cfgfile, char *weightfile, int gpu_id=0,float nms = 0.4, float thresh = 0.4, float hier_thresh = 0.5);

    ~yoloDetector();

    /**
     * yolo's forward function
     * @param img input img
     * @param det results struct which store rect of object detection, scores and id(0: front head; 1: back head; 2: bags
     */
    void detector(cv::Mat &img, std::vector<det_class> &det);

    /**
     * multiple targets in one image
     * @param img input image
     * @param out multiple targets detected
     */
    void process(cv::Mat &img, std::vector<det_class> &out);

    // this interface will delete
    void humanprocess(cv::Mat &img, std::vector<det_class> &out);

private:
    int max_idx(std::vector<float> &f);

    float iou(det_class &det1, det_class &det2);

    void non_maxima_suppression(std::vector<det_class> &det, float fth);

private:
    float _nms;
    float _thresh;
    float _hier_thresh;
    char *_cfgfile;
    char *_weightfile;
    layer _l;
    network* _net;
};


#endif // OBJECT_DETECTOR_H
