//
// Created by xulishuang on 18-1-23.
//

#include "CrowdEstimate.h"
#include "../mask_rcnn/frcnn_utils.hpp"


CrowdEstimate::CrowdEstimate()
{
    netName_ = "./models/crowd.prototxt";
    modelName_="./models/crowd_iter_1124000.caffemodel";
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);
    //net_.reset(new caffe::Net<float>(netName_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    //net_->CopyTrainedLayersFrom(modelName_);
    net_.reset(new caffe::Net<float>(netName_,caffe::TEST));
    net_->CopyTrainedLayersFrom(modelName_);
}

CrowdEstimate &CrowdEstimate::ins()
{
    static thread_local CrowdEstimate obj;
    return obj;
}


void CrowdEstimate::process(const cv::Mat src, cv::Mat &dst,float &headCount)
{
//    double start = cv::getTickCount();
    if(src.empty())
    {
        return;
    }

    float scale_factor = caffe::Frcnn::get_scale_factor(src.cols, src.rows, 600, 1000);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(), scale_factor, scale_factor);
    cv::Mat x;
    cv::resize(resized,x,cv::Size(resized.cols/4*4,resized.rows/4*4));
    auto &input_layer = net_->input_blobs()[0];
    input_layer->Reshape({1,1,x.rows,x.cols});
    int w = input_layer->width();
    int h = input_layer->height();
    cv::resize(x,x,cv::Size(w,h),0,0,cv::INTER_CUBIC);
    //net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();
    std::vector<cv::Mat> input_channels;
    for(int i =0;i<input_layer->shape()[1];i++)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->width()*input_layer->height();
    }

    if(x.channels() == 3)
    {
        cv::cvtColor(x,x,CV_BGR2GRAY);
    }
    cv::Mat floatImg;
    x.convertTo(floatImg,CV_32FC1);
    cv::split(floatImg,input_channels);
    net_->Forward();

    caffe::shared_ptr<caffe::Blob<float> > res = net_->blob_by_name("fuse_conv");
    const float *data = res->cpu_data();
    cv::Mat gray = cv::Mat(res->shape(2),res->shape(3),CV_8UC1);


    //compute the total count of heads
    float max = std::numeric_limits<float>::lowest();

    for(int i = 0;i < gray.rows * gray.cols; i++)
    {
        if(max < data[i])
        {
            max = data[i];
        }

        headCount += data[i];
    }

    LOG(INFO) << "res: " << headCount;
    unsigned char *pData = gray.data;
    for(int i = 0;i < gray.rows * gray.cols; i++)
    {
        pData[i] = data[i] * 255 /max;
    }

    dst = gray.clone();
//    falColor(gray,dst);
    /*****For Debug*****/

//    float pTest[gray.rows * gray.cols ];
//    for(int i = 0;i < gray.rows * gray.cols; i++)
//    {
//        pTest[i]=data[i];
//    }

//    cv::Mat dst2 = cv::Mat(dst.rows,dst.cols,CV_8UC1,pData);
//    cv::imshow("hotttt", dst2);
//    cv::waitKey(0);
//    std::cout << std::endl;
//    std::cout << dst2 << std::endl;
//    std::cout << std::endl;

//    double end = cv::getTickCount();
//    LOG(INFO) << modelName_ << "single img cal time: [ " << (end-start)*1000/cv::getTickFrequency() << "ms ]";
//    LOG(INFO) << std::endl;

}
