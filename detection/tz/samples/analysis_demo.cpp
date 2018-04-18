#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "../include/tuzhenalginterface.hpp"
#include <boost/thread.hpp>
#include "PutChineseToMat.h"
#include <opencv/cxcore.h>
#include <cairo/cairo-svg.h>


int test_multiVideos(void);
int test_feacompare(void);
int test_default(void);

int show_tuzhenresult(vector<PersonInfo> &personvec, Mat originImg, Mat showAttrImg, int frameNum);
int show_tuzhenresult_vehicle(vector<VehicleInfo>&vehicles,Mat img);

std::vector<cv::Scalar> colors_;



void analysis_thread1(int device){
    //cv::VideoCapture cap("/media/d/datasets/tuzhen_video/20170818.mp4");
    //cv::VideoCapture cap("./test.avi");
    cv::VideoCapture cap("./1.mp4");
    cv::Mat frame;

    //cv::VideoWriter write;
    //write.open("/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/result.avi", CV_FOURCC('M','J','P','G'), 25, cv::Size(1920, 1080));

    void *pvTuzhenHandle=NULL;
    //get para from configure file
    TuzhenPara para;
    para.deviceID = device;
    para.intervalFrames = 25;
    //para.eoptfun = EM_OPTIONAL_ITEM_PEDESTRAIN;
//      para.eoptfun = EM_OPTIONAL_ITEM_VEHICLE;
    para.eoptfun = EM_OPTIONAL_ITEM_HYBRID;
    para.mask_rcnn_prob_thresh = 0.9;
    para.face_min_height = 40;
    para.face_min_width = 50;
//    para.eoptfun = EM_OPTIONAL_ITEM_CROWD_ESTIMATE;

    boost::posix_time::ptime time_now,time_now1;
    boost::posix_time::millisec_posix_time_system_config::time_duration_type time_elapse;

    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
    if(iRtn != EM_SUCESS_STATE)
        return ;

    TuzhenInput tzInput;
    TuzhenOutput tzOutput;

    int frameNum = 0;

    while(true)
    {
        cap >> frame;
        if(frame.empty()) break;

        frameNum++;
        if(frameNum % 25 != 1)
            continue;

        tzInput.cameraID = 0;
        tzInput.img = frame;
        tzInput.sourceType = EM_SOURCE_VIDEO;
        tzOutput.personvec.clear();
        time_now=boost::posix_time::microsec_clock::universal_time();
        tuzhen_process(pvTuzhenHandle, tzInput, tzOutput);

        time_now1= boost::posix_time::microsec_clock::universal_time();

        time_elapse = time_now1 - time_now;

        std::cout << "t1 time per frame: " << time_elapse.total_milliseconds() << std::endl;

        TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
        Mat showImage = frame.clone();
        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_PEDESTRAIN){

            show_tuzhenresult(tzOutput.personvec, frame, showImage, frameNum);
            //break;
            //cv::imshow("result", frame);
            //cv::imshow("Attr", showImage);
            cv::imshow("Attr", showImage);
        }
        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_VEHICLE){
            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,frame);
            show_tuzhenresult_vehicle(tzOutput.vehicles_info,showImage);
            //cv::imshow("Attr", showImage);
            //break;
            //cv::imshow("result", frame);
            cv::imshow("Attr", showImage);
        }
        if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_CROWD_ESTIMATE){
            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,frame);
            //show_tuzhenresult_vehicle(tzOutput.vehicles_info,showImage);
            //break;
            //cv::imshow("result", frame);
            cv::imshow("Attr", showImage);
            cv::imshow("Attr", tzOutput.crowd_estimate_info.heat_map);

        }

        /// write << frame;
        cv::waitKey(50);

//        char c = cv::waitKey(0);
//        if(c == 27)
//        {
//            tuzhen_close(pvTuzhenHandle);
//            break;
//        }
    }

    return ;

}


void analysis_thread2(int device){
    //cv::VideoCapture cap("/media/d/datasets/tuzhen_video/20170818.mp4");
    cv::VideoCapture cap("./plate_test.avi");
    cv::Mat frame;

    //cv::VideoWriter write;
    //write.open("/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/result.avi", CV_FOURCC('M','J','P','G'), 25, cv::Size(1920, 1080));

    boost::posix_time::ptime time_now,time_now1;
    boost::posix_time::millisec_posix_time_system_config::time_duration_type time_elapse;

    void *pvTuzhenHandle;
    //get para from configure file
    TuzhenPara para;
    para.deviceID = device;
    para.intervalFrames = 25;
    //para.eoptfun = EM_OPTIONAL_ITEM_FACE_DETECT | EM_OPTIONAL_ITEM_CROWD_DETECT;

    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
    if(iRtn != EM_SUCESS_STATE)
        return ;

    TuzhenInput tzInput;
    TuzhenOutput tzOutput;

    int frameNum = 0;

    while(true)
    {
        cap >> frame;
        if(frame.empty()) break;

        frameNum++;
        if(frameNum % 25 != 1)
            continue;

        tzInput.cameraID = 0;
        tzInput.img = frame;
        tzOutput.personvec.clear();
        time_now=boost::posix_time::microsec_clock::universal_time();
        tuzhen_process(pvTuzhenHandle, tzInput, tzOutput);


        time_now1= boost::posix_time::microsec_clock::universal_time();

        time_elapse = time_now1 - time_now;

        std::cout << "t2 time per frame: " << time_elapse.total_milliseconds() << std::endl;
        Mat showImage = frame.clone();
        show_tuzhenresult(tzOutput.personvec, frame, showImage, frameNum);

        //cv::imshow("result", frame);
        //cv::imshow("Attr", showImage);
        /// write << frame;
        cv::waitKey(1);
//
//        char c = cv::waitKey(0);
//        if(c == 27)
//        {
    //            tuzhen_close(pvTuzhenHandle);
//            break;
//        }
    }

    return ;

}

int main(void)
{

    //select which function to test
    int sltfun = 0;
    std::cout << "Tuzhen Project Test Begin............................" << std::endl;

    colors_.push_back(cv::Scalar(0, 255, 255));
    colors_.push_back(cv::Scalar(0, 255, 0));
    colors_.push_back(cv::Scalar(255, 255, 0));
    colors_.push_back(cv::Scalar(255, 0, 255));
    colors_.push_back(cv::Scalar(0, 0, 255));
    colors_.push_back(cv::Scalar(255, 125, 0));
    colors_.push_back(cv::Scalar(125, 255, 0));
    colors_.push_back(cv::Scalar(0, 255, 125));
    colors_.push_back(cv::Scalar(0, 125, 255));
    colors_.push_back(cv::Scalar(255, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 255));
    colors_.push_back(cv::Scalar(125, 125, 255));
    colors_.push_back(cv::Scalar(125, 255, 125));
    colors_.push_back(cv::Scalar(255, 125, 125));
    colors_.push_back(cv::Scalar(0, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 0));
    colors_.push_back(cv::Scalar(0, 125, 0));

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
        case 1:
            test_feacompare();
            break;
        default:
            test_default();
            break;
    }

    return 0;
}

int test_multiVideos(void)
{
    //cv::VideoCapture cap("/media/d/datasets/tuzhen_video/20170818.mp4");
    cv::VideoCapture cap("/hard_disk2/街道行人视频拍摄/户外拍摄采集_20171213/hu/‏‎1002.MOV");
    cv::Mat frame;

    //cv::VideoWriter write;
    //write.open("/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/result.avi", CV_FOURCC('M','J','P','G'), 25, cv::Size(1920, 1080));

    void *pvTuzhenHandle;
    //get para from configure file
    TuzhenPara para;
    para.deviceID = 0;
    para.intervalFrames = 25;
    //para.eoptfun = EM_OPTIONAL_ITEM_FACE_DETECT | EM_OPTIONAL_ITEM_CROWD_DETECT;
    para.eoptfun =EM_OPTIONAL_ITEM_PEDESTRAIN;

    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
    if(iRtn != EM_SUCESS_STATE)
        return -1;

    TuzhenInput tzInput;
    TuzhenOutput tzOutput;

    int frameNum = 0;

    while(true)
    {
        cap >> frame;
        if(frame.empty()) break;

        frameNum++;
        if(frameNum % 25 != 1)
            continue;

        tzInput.cameraID = 0;
        tzInput.img = frame;
        tzOutput.personvec.clear();
        tuzhen_process(pvTuzhenHandle, tzInput, tzOutput);


        Mat showImage = frame.clone();
        show_tuzhenresult(tzOutput.personvec, frame, showImage, frameNum);


        cv::imshow("result", frame);
        cv::imshow("Attr", showImage);
       /// write << frame;
        cv::waitKey(1);

        char c = cv::waitKey(0);
        if(c == 27)
        {
            tuzhen_close(pvTuzhenHandle);
            break;
        }
    }

    return 0;
}

int test_feacompare(void)
{
    //step1:
    cv::VideoCapture cap("/media/zht/ed3258c3-8c22-4164-bfc4-e97326416a12/video/C0045奥体中心地铁A出口_2017-08-07_09h00min26s.csv5");
    cv::Mat frame;
    void *pvFeaCompare;

    return 0;
}

int test_default(void)
{
    cv::VideoCapture cap("/media/zht/ed3258c3-8c22-4164-bfc4-e97326416a12/video/C0045奥体中心地铁A出口_2017-08-07_09h00min26s.csv5");
    cv::Mat frame;
    while(true)
    {
        cap >> frame;
        cv::imshow("default", frame);
        char c = cv::waitKey(1);
        if (c == 27)
            break;
    }
    return 0;
}

std::map<int,string > lenghtMap={std::pair<int,string>(0,"long"),std::pair<int,string>(1,"short"),std::pair<int,string>(-1,"unknow"), std::pair<int,string>(-100,"undo")};
//
//std::map<int,string > colorMap={std::pair<int,string>(0,"black"),std::pair<int,string>(1,"white"),
//                                             std::pair<int,string>(2,"gray"),std::pair<int,string>(3,"red"),
//                                             std::pair<int,string>(4,"pink"),std::pair<int,string>(5,"brown"),
//                                             std::pair<int,string>(6,"orange"),std::pair<int,string>(7,"yellow"),
//                                             std::pair<int,string>(8,"green"),std::pair<int,string>(9,"cyan"),
//                                             std::pair<int,string>(10,"blue"),std::pair<int,string>(11,"purple"),
//                                             std::pair<int,string>(12,"none"),std::pair<int,string>(13,"h-stripe"),
//                                             std::pair<int,string>(14,"v-stripe"),std::pair<int,string>(15,"grid"),
//                                             std::pair<int,string>(-1,"unkonw"), std::pair<int,string>(-100,"undo")};


std::map<int,string > colorMap={std::pair<int,string>(0,"black"),std::pair<int,string>(1,"white"),
                                std::pair<int,string>(2,"gray"),std::pair<int,string>(3,"red"),
                                std::pair<int,string>(8,"green"),
                                std::pair<int,string>(10,"blue"),
                                std::pair<int,string>(100,"other"),
                                std::pair<int,string>(-1,"unkonw")};

std::map<int, string> headMap = {std::pair<int, string>(EM_PERSON_HEAD_FRONT, "front"),
                                 std::pair<int, string>(EM_PERSON_HEAD_BACK, "back"),
                                 std::pair<int, string>(EM_PERSON_HEAD_UNKNOWN, "unknown")};


std::map<int, string> bagMap = {std::pair<int, string>(EM_BACKPACK_FALSE, "bag_not"),
                                std::pair<int, string>(EM_BACKPACK_TRUE, "bag"),
                                std::pair<int, string>(EM_BACKPACK_UNKNOWN, "unknown")};



std::map<int, string> bareMap = {std::pair<int, string>(EM_BARE_FALSE, "bare_not"),
                                 std::pair<int, string>(EM_BARE_TRUE, "bare"),
                                 std::pair<int, string>(EM_BARE_UNKNOWN, "unknown")};



std::map<int, string> bangMap = {std::pair<int, string>(EM_FRINGE_FALSE, "fring_not"),
                                 std::pair<int, string>(EM_FRINGE_TRUE, "fring"),
                                 std::pair<int, string>(EM_FRINGE_UNKNOWN, "unknwon")};



std::map<int, string> genderMap = {std::pair<int, string>(EM_GENDER_FEMALE, "female"),
                                   std::pair<int, string>(EM_GENDER_MALE, "male"),
                                   std::pair<int, string>(EM_GENDER_UNKNOWN, "unknown")};



std::map<int, string> glassMap = {std::pair<int, string>(EM_GLASS_FALSE, "glass_not"),
                                  std::pair<int, string>(EM_GLASS_COMMON, "common_glass"),
                                  std::pair<int, string>(EM_GLASS_SUNGLASS, "sunglass"),
                                  std::pair<int, string>(EM_GLASS_UNKNOWN, "unknown")};



std::map<int, string> hatMap = {std::pair<int, string>(EM_HAT_FALSE, "hat_not"),
                                std::pair<int, string>(EM_HAT_TRUE, "hat"),
                                std::pair<int, string>(EM_HAT_UNKNOWN, "unknown")};



std::map<int, string> maskMap = {std::pair<int, string>(EM_MASK_FALSE, "mask_not"),
                                 std::pair<int, string>(EM_MASK_TRUE, "mask"),
                                 std::pair<int, string>(EM_MASK_UNKNOWN, "unknown")};



std::map<int, string> ageMap = {std::pair<int, string>(EM_AGE_CHILD, "child"),
                                std::pair<int, string>(EM_AGE_MIDDLE_AGE, "middle"),
                                std::pair<int, string>(EM_AGE_YOUTH, "youth"),
                                std::pair<int, string>(EM_AGE_OLD, "old"),
                                std::pair<int, string>(EM_AGE_UNKNOWN, "unknown")};



int show_tuzhenresult(vector<PersonInfo> &personvec, Mat frame, Mat showAttrImg, int frameNum)
{
    char strframeNum[100], strID[100], strscore[100];
    sprintf(strframeNum, "%d", frameNum);
    int base = 10;
    int interval = 15;

    if(!personvec.empty())
    {
        //show result
        cv::putText(frame, strframeNum, cvPoint(50, 50), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
        cv::putText(showAttrImg, strframeNum, cvPoint(50, 50), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
        for(PersonInfo& person : personvec)
        {
            cv::Rect &rectone = person.personrct;
            //char strframeNum[100], strID[100], strscore[100];
            //sprintf(strframeNum, "%3.2f", person.fscore);
            //cv::putText(showAttrImg, strframeNum,  Point(rectone.tl().x, rectone.tl().y - 10), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
            if(person.breport)
            {
                cv::rectangle(frame, rectone, CV_RGB(255,0,0), 3, 8, 0);
                cv::rectangle(showAttrImg, rectone, CV_RGB(255,0,0), 3, 8, 0);
//                std::cout << "ID: " << person.objectID << std::endl;
//                std::cout << lenghtMap[person.sleeve_length] << std::endl;
//                std::cout << lenghtMap[person.pants_length] << std::endl;
                cv::putText(showAttrImg, lenghtMap[person.sleeve_length], Point(rectone.tl().x, rectone.tl().y + base), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(200,200,50));
                cv::putText(showAttrImg, lenghtMap[person.pants_length], Point(rectone.tl().x, rectone.tl().y + base + interval), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                cv::putText(showAttrImg, colorMap[person.coat_color], Point(rectone.tl().x, rectone.tl().y + base + interval*2), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                cv::putText(showAttrImg, colorMap[person.trouser_color], Point(rectone.tl().x, rectone.tl().y + base + interval*3), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));

                if(person.bgetface)
                {
                    cv::Rect bigImgRect;
                    bigImgRect.x = person.facerct.x + person.personrct.x;
                    bigImgRect.y = (person.facerct.y + person.personrct.y);
                    bigImgRect.width = person.facerct.width;
                    bigImgRect.height = person.facerct.height;

                    cv::rectangle(showAttrImg, bigImgRect, CV_RGB(0,255,255), 3, 8, 0);

                    for(Vec2f& landmark : person.face_landmarks)
                    {
                        cv::circle(showAttrImg,cv::Point(int(landmark[0])+bigImgRect.x,int(landmark[1])+bigImgRect.y),2,cv::Scalar(255,0,0));
                    }
                    cv::putText(showAttrImg, ageMap[person.age_info], Point(rectone.tl().x, rectone.tl().y + base + interval*4), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, maskMap[person.mask_info], Point(rectone.tl().x, rectone.tl().y + base + interval*5), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, hatMap[person.hat_info], Point(rectone.tl().x, rectone.tl().y + base + interval*6), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, glassMap[person.glass_info], Point(rectone.tl().x, rectone.tl().y + base + interval*7), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));

                    cv::putText(showAttrImg, genderMap[person.gender_info], Point(rectone.tl().x, rectone.tl().y + base + interval*8), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, bangMap[person.fringe_info], Point(rectone.tl().x, rectone.tl().y + base + interval*9), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, bareMap[person.bare_info], Point(rectone.tl().x, rectone.tl().y + base + interval*10), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, bagMap[person.backpack], Point(rectone.tl().x, rectone.tl().y + base + interval*11), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                    cv::putText(showAttrImg, headMap[person.head_type], Point(rectone.tl().x, rectone.tl().y + base + interval*12), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0,255,0));
                }



            }
            else
            {
                cv::rectangle(frame, rectone, CV_RGB(0,255,0), 3, 8, 0);
                cv::rectangle(showAttrImg, rectone, CV_RGB(0,255,0), 3, 8, 0);
            }
            sprintf(strID, "%d", person.objectID);
            cv::putText(frame, strID, Point(rectone.tl().x, rectone.tl().y - 5), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(0, 0, 255));
            cv::putText(showAttrImg, strID, Point(rectone.tl().x, rectone.tl().y - 5), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(0, 0, 255));

            sprintf(strscore, "%4.2f", person.fscore);
            cv::putText(showAttrImg, strscore, Point(rectone.tl().x + 10, rectone.tl().y -15), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0f, CV_RGB(255, 127, 0));

            const std::vector<unsigned int> render_pairs {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17};
            std::vector<std::pair<size_t, size_t> > matchpair;
            for (size_t i=0; i<render_pairs.size()/2; i++)
                matchpair.push_back({render_pairs[2*i], render_pairs[2*i+1]});

            //for mask rcnn
//            const std::vector<unsigned int> render_pairs{17, 6, 17, 5, 6, 8, 8, 10, 5, 7, 7, 9, 17, 12, 12, 14, 14, 16, 17,
//                                                         11, 11, 13, 13, 15, 17, 0, 0, 2, 2, 4, 0, 1, 1, 3};
//            for (size_t i = 0; i < render_pairs.size() / 2; i++)
//                matchpair.push_back({render_pairs[2 * i], render_pairs[2 * i + 1]});


            auto &pts_one=person.pts;
            if (pts_one.empty())
                continue;

//            for (int i = 0; i < 17; i++) {
//                cv::Vec3f point = pts_one[i];
//                int x = point[0] * 1.0 / 28 * person.personrct.width;
//                int y = point[1] * 1.0 / 28 * person.personrct.height;
//                pts_one[i][0] = x +  person.personrct.x;
//                pts_one[i][1] = y + person.personrct.y;
//            }
//            cv::Vec3f point_neck;
//            point_neck[0] = (pts_one[5][0] + pts_one[6][0]) / 2;
//            point_neck[1] = (pts_one[5][1] + pts_one[6][1]) / 2;
//            point_neck[2] = (pts_one[5][2] + pts_one[6][2]) / 2;
//            pts_one.push_back(point_neck);

            auto f=[&](int idpart, const cv::Scalar &color)
            {
                std::pair<size_t, size_t> &bodypart=matchpair[idpart];
                if (std::min(pts_one[bodypart.first][2], pts_one[bodypart.second][2]) < 0.1)
                    return;
                int id1=bodypart.first;
                int id2=bodypart.second;
                cv::line(showAttrImg,
                         cv::Point(pts_one[id1][0], pts_one[id1][1]),
                         cv::Point(pts_one[id2][0], pts_one[id2][1]),
                         color, 2);
            };
            for (size_t idpart=0; idpart<matchpair.size(); idpart++)
            {
                f(idpart, colors_[idpart]);
            }

        }
    }
    return 0;
}

void putTextCairo(cv::Mat &targetImage, std::string const& text,
        cv::Point2d centerPoint, std::string const& fontFace,
        double fontSize, cv::Scalar textColor, bool fontItalic, bool fontBold)
{
    // Create Cairo
    cairo_surface_t* surface = cairo_image_surface_create(
                CAIRO_FORMAT_ARGB32,
                targetImage.cols,
                targetImage.rows);

    cairo_t* cairo = cairo_create(surface);

    // Wrap Cairo with a Mat
    cv::Mat cairoTarget(cairo_image_surface_get_height(surface),
                cairo_image_surface_get_width(surface),
                CV_8UC4,
                cairo_image_surface_get_data(surface),
                cairo_image_surface_get_stride(surface));

    // Put image onto Cairo
    cv::cvtColor(targetImage, cairoTarget, cv::COLOR_BGR2BGRA);

    // Set font and write text
    cairo_select_font_face (cairo,
                fontFace.c_str(),
                fontItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL,
                fontBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);

    cairo_set_font_size(cairo, fontSize);
    cairo_set_source_rgb(cairo, textColor[2], textColor[1], textColor[0]);

    cairo_text_extents_t extents;
    cairo_text_extents(cairo, text.c_str(), &extents);

    cairo_move_to(cairo,
                centerPoint.x - extents.width/2 - extents.x_bearing,
                centerPoint.y - extents.height/2- extents.y_bearing);
    cairo_show_text(cairo, text.c_str());

    // Copy the data to the output image
    cv::cvtColor(cairoTarget, targetImage, cv::COLOR_BGRA2BGR);
    cairo_destroy(cairo);
    cairo_surface_destroy(surface);
}

int show_tuzhenresult_vehicle(vector<VehicleInfo>&vehicles,Mat img) {

    if (!vehicles.empty()) {
        double size = (double)MIN(20, 20);
        string font_path = "./Font/msyh.ttf";
        //show result
        //cv::putText(frame, strframeNum, cvPoint(50, 50), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
        //cv::putText(showAttrImg, strframeNum, cvPoint(50, 50), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
        for (VehicleInfo &vehicle : vehicles) {
            cv::Rect &rectone = vehicle.vehicle_location;
            if(vehicle.breport){
                cv::rectangle(img, rectone, CV_RGB(255,0,0), 3, 8, 0);
            }else{
                cv::rectangle(img, rectone, CV_RGB(0,255,0), 3, 8, 0);
            }
            //std::cout << vehicle.vehicle_brand << vehicle.vehicle_color << vehicle.card_no << vehicle.card_color << endl;
            //PutText_2_IplImg(img, vehicle.vehicle_brand.c_str(), Point(rectone.tl().x, rectone.tl().y - 5), Scalar(0, 0, 255), 20, 0.2);
            //PutText_2_IplImg(img, vehicle.vehicle_color.c_str(), Point(rectone.tl().x, rectone.tl().y + 5), Scalar(0, 0, 255), 20, 0.2);
            //PutText_2_IplImg(img, vehicle.card_no.c_str(), Point(rectone.tl().x, rectone.tl().y + 15), Scalar(0, 0, 255), 20, 0.2);
            //PutText_2_IplImg(img, vehicle.card_color.c_str(), Point(rectone.tl().x, rectone.tl().y + 25), Scalar(0, 0, 255), 20, 0.2);

            putTextCairo(img, vehicle.vehicle_brand.c_str(), Point(rectone.tl().x, rectone.tl().y - 5), font_path, size, Scalar(0,0,255), false, false);
            putTextCairo(img, vehicle.vehicle_color.c_str(), Point(rectone.tl().x, rectone.tl().y + 15), font_path, size, Scalar(0,0,255), false, false);
            putTextCairo(img, vehicle.card_no.c_str(), Point(rectone.tl().x, rectone.tl().y + 35), font_path, size, Scalar(0,0,255), false, false);
            putTextCairo(img, vehicle.card_color.c_str(), Point(rectone.tl().x, rectone.tl().y + 55), font_path, size, Scalar(0,0,255), false, false);

            char strframeNum[100], strID[100], strscore[100];
            sprintf(strframeNum, "%3.2f", vehicle.fscore);
            cv::putText(img, strframeNum,  Point(rectone.tl().x, rectone.tl().y - 10), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(255, 0, 0));
            sprintf(strID, "%d", vehicle.objectID);
            cv::putText(img, strID, Point(rectone.tl().x, rectone.tl().y - 5), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(0, 0, 255));

        }
    }
}
