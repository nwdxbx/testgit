//
// Created by xulishuang on 18-2-26.
//
#include "../source/reIDfeature/face_re_id.hpp"

int main(void)
{

    //select which function to test
    //int sltfun = 0;
//    std::cout << "Tuzhen Feature Cal Test Begin............................" << std::endl;
//    while(true);
    //person_reid_test();
    //face_reid_test();
    std::cout << "test mode....";
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);

    PersonInfo personInfo;
    std::vector<float> facefeat;
    cv::Mat frame = cv::imread("./face_landmark_test.bmp");
    Face_Re_ID::ins().process(frame, frame, cv::Point(0, 0), facefeat,personInfo);
    for(Vec2f& landmark : personInfo.face_landmarks)
    {
        cv::circle(frame,cv::Point(int(landmark[0]),int(landmark[1])),2,cv::Scalar(255,0,0));
    }
    //cv::rectangle(frame, bigImgRect, CV_RGB(0,255,255), 3, 8, 0);
    cv::imshow("Attr", frame);
    cv::waitKey(0);
    return 0;
}