#include <iostream>
#include "ssd.hpp"

using namespace std;

int main()
{
    std::ifstream flist("./test.txt");
    std::string imgname;
    ssd_detect_caffe::ins();
    while(flist >>imgname)
    {
        cv::Mat img=cv::imread(imgname);
        ssd_detres res;
        ssd_detect_caffe::ins().process(img,res);
        cv::rectangle(img,cv::Rect(res.x1,res.y1,res.x2-res.x1,res.y2-res.y1),cv::Scalar(255,0,255),2,8);
        std::cout<<"label:"<<res.label<<std::endl;
        cv::imshow("img",img);
        cv::waitKey(0);
    }
    cout << "Hello World!" << endl;
    return 0;
}
