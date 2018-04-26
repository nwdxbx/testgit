//
// Created by xulishuang on 17-11-24.
//

#include "trackNet.h"


trackNet &trackNet::ins()
{
    static thread_local trackNet obj;
    return obj;
}


trackNet::trackNet()
{
    netName_ = "./models/track_p";
    modelName_="./models/track_c";

    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);

    net_.reset(new caffe::Net<float>(netName_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(modelName_);

    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();

}

void trackNet::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
}

void trackNet::process(cv::Mat &imgin, std::vector<float> &result)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(imgin);
    std::vector<std::vector<float>> res;
    process(pkgs, res);
    result=res[0];
}

void trackNet::process(std::vector< cv::Mat> &imgin, std::vector<std::vector<float>> &result)
{
    //std::cout << "\t\tthe size of the network is (" << w_ << "x" << h_ << ")" << std::endl;

    auto &input_layer=net_->input_blobs()[0];
    input_layer->Reshape({(int)imgin.size(), input_layer->shape(1),
                          input_layer->shape(2), input_layer->shape(3)});
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();


    //auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();

    std::vector<std::vector<cv::Mat>>  input_channels;
    input_channels.resize(input_layer->shape()[0]);
    for(int i=0;i<input_layer->shape()[0];i++)
    {
        for (int j=0; j<input_layer->shape()[1]; j++)
        {
            cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
            input_channels[i].push_back(channel);
            input_data += input_layer->width()*input_layer->height();
        }

        cv::Mat x=imgin[i].clone();
        if(x.empty())
            x=cv::Mat(w_,h_,CV_8UC3, cv::Scalar::all(0));
        else
            cv::resize(x, x, cv::Size(w_, h_), (0, 0), (0, 0), cv::INTER_CUBIC);
        //cv::resize(x, x, cv::Size(w_, h_));

        if (x.channels()==1)
            cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);
        cv::Mat floatImg, floatImg1;
        x.convertTo(floatImg1, CV_32FC3);
        preprocess(floatImg1, floatImg);
        cv::split(floatImg, input_channels[i]);

    }
    net_->Forward();
    get_res(net_->output_blobs(), result);

}

void trackNet::get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res)
{
    const float *resk=net_res[0]->cpu_data();
    res.resize(net_res[0]->shape(0));
    for (int i=0; i<net_res[0]->shape(0); i++)
    {
        res[i].clear();
        res[i].insert(res[i].begin(), resk, resk+(net_res[0]->shape(1)));
        resk+=net_res[0]->shape(1);
    }
}