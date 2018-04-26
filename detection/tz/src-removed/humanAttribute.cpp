#include "humanAttribute.hpp"
#include <glog/logging.h>

std::map<int,int> colorMap={std::pair<int,int>(0,0),std::pair<int,int>(1,1),std::pair<int,int>(2,2),std::pair<int,int>(3,3),std::pair<int,int>(4,8),
                            std::pair<int,int>(5,10),std::pair<int,int>(6,-1)};

humanAttribute_caffe &humanAttribute_caffe::ins()
{
    static thread_local humanAttribute_caffe obj;
    return obj;
}

humanAttribute_caffe::humanAttribute_caffe()
    : netname1_("./models/humanAttribute_p"),
      modelname1_("./models/humanAttribute_c"),
      netname2_("./models/color_p"),
      modelname2_("./models/color_c")
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);
    net1_.reset(new caffe::Net<float>(netname1_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net1_->CopyTrainedLayersFrom(modelname1_);
    net2_.reset(new caffe::Net<float>(netname2_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net2_->CopyTrainedLayersFrom(modelname2_);
}


void humanAttribute_caffe::process(cv::Mat &imgin,det_res &det)
{
    auto &input_layer=net1_->input_blobs()[0];
    cv::Mat sample_single;
    cv::Mat m=imgin.clone();
    cv::resize(m,m,cv::Size(input_layer->width(),input_layer->height()));
    m.convertTo(sample_single,CV_32FC3);
    cv::Scalar mean=cv::Scalar::all(127.5);
    sample_single -=mean;
    sample_single *=0.0078125;
    cv::Mat color_single = sample_single.clone();
    float *input_data=NULL;
    input_data=input_layer->mutable_cpu_data();
    std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
    for(int i=0;i<input_layer->channels();i++)
    {
        cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
        input_channels->push_back(channel);
        input_data +=(input_layer->height())*(input_layer->width());
    }
    cv::split(sample_single,*input_channels);
    net1_->Forward();
    auto &sleeve_blob = net1_->blob_by_name("loss3/sleeve/prob");
    auto &pants_blob = net1_->blob_by_name("loss3/pants/prob");
    const float* sleeve_data=NULL;
    const float* pants_data=NULL;
    sleeve_data=(const float*)sleeve_blob->cpu_data();
    pants_data=(const float*)pants_blob->cpu_data();
    if(sleeve_data[0]>=sleeve_data[1]){
        det.sleeveLength=0;
        det.sleeveLengthScore = sleeve_data[0];
    }
//    if(sleeve_data[0]<sleeve_data[1] && sleeve_data[1]>0.5){
//        det.sleeveLength=1;
//        det.sleeveLengthScore = sleeve_data[1];
//    }
    else{
        det.sleeveLength=1;
        det.sleeveLengthScore = sleeve_data[1];
    }
    if(pants_data[0]>=pants_data[1]){
        det.pantsLength = 0;
        det.pantsLengthScore = pants_data[0];
    }
//    if(pants_data[0]<pants_data[1] && pants_data[1]>0.5){
//        det.pantsLength=1;
//        det.pantsLengthScore = pants_data[1];
//    }
    else{
        det.pantsLength=1;
        det.pantsLengthScore = pants_data[1];
    }


    auto &color_input_layer = net2_->input_blobs()[0];
    float *color_input_data=NULL;
    color_input_data=color_input_layer->mutable_cpu_data();
    std::vector<cv::Mat>* color_input_channels = new std::vector<cv::Mat>;
    for(int i=0;i<color_input_layer->channels();i++)
    {
        cv::Mat channel(color_input_layer->height(),color_input_layer->width(),CV_32FC1,color_input_data);
        color_input_channels->push_back(channel);
        color_input_data +=(color_input_layer->height())*(color_input_layer->width());
    }

    cv::split(color_single,*color_input_channels);
    net2_->Forward();
    auto &up_blob = net2_->blob_by_name("up_lossColor/prob");
    auto &down_blob = net2_->blob_by_name("down_lossColor/prob");
    const float* up_data=NULL;
    const float* down_data=NULL;
    up_data=(const float*)up_blob->cpu_data();
    down_data=(const float*)down_blob->cpu_data();

    std::vector<std::pair<float,int> >up_pairs;
    std::vector<std::pair<float,int> >down_pairs;
    for(size_t i=0;i<up_blob->channels();i++)
        up_pairs.push_back(std::pair<float,int>(up_data[i],i));
    std::partial_sort(up_pairs.begin(),up_pairs.begin()+up_blob->channels(),up_pairs.end(),[](const std::pair<float,int>& lhs,const std::pair<float,int>& rhs)
    {
            return lhs.first>rhs.first;
    });
    for(size_t i=0;i<down_blob->channels();i++)
        down_pairs.push_back(std::pair<float,int>(down_data[i],i));
    std::partial_sort(down_pairs.begin(),down_pairs.begin()+down_blob->channels(),down_pairs.end(),[](const std::pair<float,int>& lhs,const std::pair<float,int>& rhs)
    {
            return lhs.first>rhs.first;
    });

    det.upclsColor = colorMap[up_pairs[0].second];
    det.coat_color_score =up_pairs[0].first;
    det.downclsColor = colorMap[down_pairs[0].second];
    det.trouser_color_score = down_pairs[0].first;

//    if(det.upclsColor!=0 && det.upclsColor!=-1)
//        if(det.coat_color_score < 0.8)
//            det.upclsColor = -1;
//    if(det.downclsColor!=0 && det.downclsColor!=-1)
//        if(det.trouser_color_score<0.9)
//            det.downclsColor = -1;

}
