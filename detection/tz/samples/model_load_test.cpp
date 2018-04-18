//
// Created by xulishuang on 17-11-24.
//

#include <caffe/caffe.hpp>
#include <iostream>

void loadModelTest(std::string proto,std::string model)
{
    boost::shared_ptr< caffe::Net<float> > net_ = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(proto, caffe::TEST));
    net_->CopyTrainedLayersFrom(model);
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
    loadModelTest("./models/track.prototxt","./models/track.caffemodel");
#ifdef NDEBUG
    //cv::rectangle(img,rects[max_score_index],cv::Scalar(255,0,0),3);
    std::cout << "release mode";
#endif
    return 0;
}