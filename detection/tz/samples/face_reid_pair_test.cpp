//
// Created by xulishuang on 18-2-26.
//

#include "../source/reIDfeature/face_re_id.hpp"


float similarity(std::vector<float>&attr1,std::vector<float>&attr2)
{
    float result =0.0,numerator=0,denominator_1=0,denominator_2=0;
    for(int i = 0; i < attr1.size();i++)
    {
        numerator += attr1[i]*attr2[i];
        denominator_1 += attr1[i]*attr1[i];
        denominator_2 += attr2[i]*attr2[i];
    }
    result = numerator/(sqrt(denominator_1)*sqrt(denominator_2));
}


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
    std::vector<float> facefeat1;
    std::vector<float> facefeat2;
    cv::Mat frame = cv::imread("./face_test1.bmp");
    cv::Mat frame2 = cv::imread("./face_test2.bmp");
    Face_Re_ID::ins().process(frame, frame, cv::Point(0, 0), facefeat1,personInfo);
    Face_Re_ID::ins().process(frame2, frame2, cv::Point(0, 0), facefeat2,personInfo);

    float simi = similarity(facefeat1,facefeat2);
    std::cout << simi << endl;

    return 0;
}