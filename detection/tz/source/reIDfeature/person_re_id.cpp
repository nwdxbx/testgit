#include "person_re_id.hpp"
#define RESIZE_WITH 48
#define RESIZE_HEIGHT 128
Person_Re_ID &Person_Re_ID::ins()
{
    static thread_local Person_Re_ID obj("./models/PersonReID_p", "./models/PersonReID_c",0);
    return obj;
}

Person_Re_ID::Person_Re_ID(const std::string &net_file,
                           const std::string &model_file, int gpuid)
                       : netname_(net_file),
                       modelname_(model_file)
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(gpuid);
    net_.reset(new caffe::Net<float>(net_file, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(model_file);

    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();
}
void Person_Re_ID::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
}
void Person_Re_ID::process(std::vector< cv::Mat> &imgin, std::vector<std::vector<float>> &result)
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
void Person_Re_ID::process(cv::Mat &imgin, std::vector<float> &result)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(imgin);
    std::vector<std::vector<float>> res;
    process(pkgs, res);
    result=res[0];
}
void Person_Re_ID::get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res)
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
