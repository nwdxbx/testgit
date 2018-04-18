//
// Created by xulishuang on 18-2-8.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "../include/tuzhenalginterface.hpp"
#include <boost/thread.hpp>
#include "PutChineseToMat.h"
#include <opencv/cxcore.h>
#include <cairo/cairo-svg.h>

void analysis_thread1(int device){
    //cv::VideoCapture cap("/media/d/datasets/tuzhen_video/20170818.mp4");
    //cv::VideoCapture cap("./test.avi");
    //cv::Mat frame;

    //cv::VideoWriter write;
    //write.open("/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/result.avi", CV_FOURCC('M','J','P','G'), 25, cv::Size(1920, 1080));

    void *pvTuzhenHandle=NULL;
    //get para from configure file
    TuzhenPara para;
    para.deviceID = device;
    para.intervalFrames = 25;
    para.eoptfun = EM_OPTIONAL_ITEM_PEDESTRAIN;
//      para.eoptfun = EM_OPTIONAL_ITEM_VEHICLE;
//    para.eoptfun = EM_OPTIONAL_ITEM_HYBRID;
    para.itvreportmove = 3;
    para.itvreportstatic = 3;
    para.mask_rcnn_prob_thresh = 0.9;
    para.face_min_height = 80;
    para.face_min_width = 60;
//    para.eoptfun = EM_OPTIONAL_ITEM_CROWD_ESTIMATE;

    boost::posix_time::ptime time_now,time_now1;
    boost::posix_time::millisec_posix_time_system_config::time_duration_type time_elapse;

    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
    if(iRtn != EM_SUCESS_STATE)
        return ;

    TuzhenInput tzInput;
    TuzhenOutput tzOutput;

    int frameNum = 0;
    cv::Mat frame = cv::imread("./suspect_person/xx5.jpg");
    tzInput.cameraID = 0;
    tzInput.img = frame;
    tzInput.sourceType = EM_SOURCE_VIDEO;
    tzOutput.personvec.clear();
    PersonInfo person;
    time_now=boost::posix_time::microsec_clock::universal_time();
    face_feature_attri(frame,person);
    cv::Mat frame0 = cv::imread("./suspect_person/xx6.jpg");
    face_feature_attri(frame0,person);
    cv::Mat frame1 = cv::imread("./suspect_person/xx7.jpg");
    face_feature_attri(frame1,person);
    cv::Mat frame2 = cv::imread("./suspect_person/xx10.jpg");
    face_feature_attri(frame2,person);
    cv::Mat frame3 = cv::imread("./suspect_person/xx12.jpg");
    face_feature_attri(frame3,person);
    cv::Mat frame4 = cv::imread("./suspect_person/xx13.jpg");
    face_feature_attri(frame4,person);
    cv::Mat frame5 = cv::imread("./suspect_person/xx14.jpg");
    face_feature_attri(frame5,person);


//    tuzhen_process(pvTuzhenHandle, tzInput, tzOutput);


//    while(true)
//    {
//        cap >> frame;
//        if(frame.empty()) break;
//
//        frameNum++;
//        if(frameNum % 25 != 1)
//            continue;
//
//        tzInput.cameraID = 0;
//        tzInput.img = frame;
//        tzInput.sourceType = EM_SOURCE_VIDEO;
//        tzOutput.personvec.clear();
//        time_now=boost::posix_time::microsec_clock::universal_time();
//        tuzhen_process(pvTuzhenHandle, tzInput, tzOutput);
//
//        time_now1= boost::posix_time::microsec_clock::universal_time();
//
//        time_elapse = time_now1 - time_now;
//
//        std::cout << "t1 time per frame: " << time_elapse.total_milliseconds() << std::endl;
//
//        TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
//        Mat showImage = frame.clone();
//        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_PEDESTRAIN){
//
//            show_tuzhenresult(tzOutput.personvec, frame, showImage, frameNum);
//            //break;
//            //cv::imshow("result", frame);
//            //cv::imshow("Attr", showImage);
//            cv::imshow("Attr", showImage);
//        }
//        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_VEHICLE){
//            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,frame);
//            show_tuzhenresult_vehicle(tzOutput.vehicles_info,showImage);
//            //cv::imshow("Attr", showImage);
//            //break;
//            //cv::imshow("result", frame);
//            cv::imshow("Attr", showImage);
//        }
//        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_CROWD_ESTIMATE){
//            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,frame);
//            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,showImage);
//            //break;
//            //cv::imshow("result", frame);
//            cv::imshow("Attr", showImage);
//            cv::imshow("Attr", tzOutput.crowd_estimate_info.heat_map);
//
//        }
//        cv::waitKey(100);
//    }

    return ;

}


int main(void)
{

    //select which function to test
    int sltfun = 0;
    std::cout << "Tuzhen Project Test Begin............................" << std::endl;

    switch(sltfun)
    {
        case 0:
            //test_multiVideos();
        {
            boost::thread t1(&analysis_thread1,0);
            //boost::thread t2(&analysis_thread2,1);
            t1.join();
            //t2.join();
            break;
        }
    }

    return 0;
}