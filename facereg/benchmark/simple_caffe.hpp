#ifndef TUZHEN_ALG_SIMPLE_CAFFE_HPP
#define TUZHEN_ALG_SIMPLE_CAFFE_HPP
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>

namespace alg
{
template <typename T>
class simple_caffe
{
public:
    using RES=T;
    using BATCH_RES=std::vector<T>;
    simple_caffe(const std::string &net_file,
            const std::string &model_file,
            int gpuid=0);

    virtual void process(cv::Mat &imgin, RES &result);
    virtual void process(std::vector< cv::Mat  > &imgin, BATCH_RES &result);
protected:
    virtual void preprocess(cv::Mat &img, cv::Mat &resimg);
    virtual void get_res(const std::vector<caffe::Blob<float>*> &net_res, BATCH_RES &res)=0;
    int w_;
    int h_;
private:
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr< caffe::Net<float> > net_;
};

template <typename T>
simple_caffe<T>::simple_caffe(const std::string &net_file,
        const std::string &model_file, int gpuid)
    : netname_(net_file),
    modelname_(model_file)
{

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(gpuid);

    net_.reset(new caffe::Net<float>(net_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(model_file);

    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();
}

template <typename T>
void simple_caffe<T>::process( std::vector< cv::Mat> &imgin,std::vector<T> & result)
{
    //std::cout << "\t\tthe size of the network is (" << w_ << "x" << h_ << ")" << std::endl;

    auto &input_layer=net_->input_blobs()[0];
    input_layer->Reshape({(int)imgin.size(), input_layer->shape(1),
            input_layer->shape(2), input_layer->shape(3)});
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();

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
            cv::resize(x, x, cv::Size(w_, h_));

        if (x.channels()==1)
            cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);
            
        cv::cvtColor(x, x, cv::COLOR_BGR2RGB);    
        cv::Mat floatImg, floatImg1;
        x.convertTo(floatImg1, CV_32FC3);

        preprocess(floatImg1, floatImg);

        cv::split(floatImg, input_channels[i]);

    }
    net_->Forward();
    get_res(net_->output_blobs(), result);

}



template <typename T>
void simple_caffe<T>::process(cv::Mat &imgin, T &result)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(imgin);
    std::vector<T> res;
    process(pkgs, res);
    result=res[0];
}

template <typename T>
void simple_caffe<T>::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    resimg=img.clone();
}
}
#endif

