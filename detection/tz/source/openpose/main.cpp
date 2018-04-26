#include <iostream>
#include <time.h>
#include "openposeCaffe.hpp"

int main()
{
    cv::Mat img= cv::imread("1.jpg");
    std::vector< cv::Mat > imgs;
    for (size_t i=0; i<8; i++)
        imgs.push_back(img);
    openpose_caffe op("./models/openpose/", property::PoseModel::MPI_15_4,
    {imgs.size(), 3, 200, 200}, img.size());
    std::vector< std::vector< std::vector<cv::Vec3f> > > pts;
    for (size_t i=0; i<10; i++)
    {
        time_t start, end;
        start=clock();
        op.process(imgs, pts);
        end=clock();
        std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
        std::cout << std::endl << std::endl;
    }
    return 0;
}
