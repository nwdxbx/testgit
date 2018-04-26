#include "ssd.hpp"
#include <glog/logging.h>

ssd_detect_caffe &ssd_detect_caffe::ins()
{
    static thread_local ssd_detect_caffe obj;
    return obj;
}

ssd_detect_caffe::ssd_detect_caffe()
    :netname_("./models/head_detection_p"),
    clsnetname_("./models/clsface_p"),
    modelname_("./models/head_detection_c"),
    clsmodelname_("./models/clsface_c")
{
    thresh_ = 0.5;  // face detector prob
    clsthresh_ = 0.7;   //bigger this thresh,then backhead
    net_.reset(new caffe::Net<float>(netname_,caffe::TEST, "B743382C96DB85858FF65AE97DF98970", "802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(modelname_);
    clsnet_.reset(new caffe::Net<float>(clsnetname_,caffe::TEST, "B743382C96DB85858FF65AE97DF98970", "802030C17462D3E3"));
    clsnet_->CopyTrainedLayersFrom(clsmodelname_);
}

void ssd_detect_caffe::clsprocess(cv::Mat &facImg, bool &flag)
{
    auto &input_layer = clsnet_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);
    float scale = 0.0087125;
    cv::Mat sample_single;
    cv::Mat m=facImg.clone();
    cv::resize(m,m,cv::Size(input_layer->width(),input_layer->height()));
    m.convertTo(sample_single,CV_32FC3);
    sample_single -=mean;
    sample_single *=scale;
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

    clsnet_->Forward();
    auto &clsface_blob = clsnet_->blob_by_name("prob");
    const float* cls_data=NULL;
    cls_data = (const float*)clsface_blob->cpu_data();
    if(cls_data[0]>clsthresh_)
        flag = false;
    else flag = true;
    std::cout<<"clsprob: "<<cls_data[0]<<std::endl;
}

void ssd_detect_caffe::process(cv::Mat &imgin,ssd_detres &res,float thresh, float clsthresh)
{
    thresh_ = thresh;
    clsthresh_ = clsthresh;
    auto &input_layer = net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);

    cv::Mat dstimage,sample;
    int height = imgin.rows;
    int width = imgin.cols;
    int abs_diff = std::abs(2*width-height);
    if(2*width<height)
    {
        cv::Rect rect(0,0,width,2*width);
        dstimage = imgin(rect).clone();
    }
    else
        cv::copyMakeBorder(imgin,dstimage,0,abs_diff,0,0,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));

    if (dstimage.channels() == 3 && num_channels == 1)
        cv::cvtColor(dstimage, sample, cv::COLOR_BGR2GRAY);
    else if (dstimage.channels() == 4 && num_channels == 1)
        cv::cvtColor(dstimage, sample, cv::COLOR_BGRA2GRAY);
    else if (dstimage.channels() == 4 && num_channels == 3)
        cv::cvtColor(dstimage, sample, cv::COLOR_BGRA2BGR);
    else if (dstimage.channels() == 1 && num_channels == 3)
        cv::cvtColor(dstimage, sample, cv::COLOR_GRAY2BGR);
    else
        sample = dstimage.clone();
    cv::resize(sample,sample,cv::Size(input_layer->width(),input_layer->height()));
    sample.convertTo(sample,CV_32FC3);
    sample -=mean;
    float *input_data=NULL;
    input_data=input_layer->mutable_cpu_data();
    std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
    for(int i=0;i<input_layer->channels();i++)
    {
        cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
        input_channels->push_back(channel);
        input_data +=(input_layer->height())*(input_layer->width());
    }
    cv::split(sample,*input_channels);

    net_->Forward();

    caffe::Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    std::vector<std::vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            result += 7;
            continue;
        }
        std::vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    std::vector<ssd_detres> r;
    r.clear();
    for(int i=0;i<detections.size();i++){
        const std::vector<float>& d=detections[i];
        const float score = d[2];
        if(score >= thresh_){
            ssd_detres det;
            det.x1 = static_cast<int>(d[3] * dstimage.cols);
            det.y1 = static_cast<int>(d[4] * dstimage.rows);
            det.x2 = static_cast<int>(d[5] * dstimage.cols);
            det.y2 = static_cast<int>(d[6] * dstimage.rows);
            det.label = static_cast<int>(d[1]);
            det.score = score;
            r.push_back(det);
        }
    }
    std::sort(r.begin(),r.end(),[](const ssd_detres &lhs,const ssd_detres &rhs){return lhs.score>rhs.score;});
    if (r.size()==0)
        res=ssd_detres{-1,-1,-1,-1,-1,-1};
    else{
        int x1,y1,x2,y2;
        if(r[0].x1<0) r[0].x1=0;
        if(r[0].y1<0) r[0].y1=0;
        if(r[0].x2>=width) r[0].x2=width-1;
        if(r[0].y2>=height) r[0].y2=height-1;
        cv::Rect rect=cv::Rect(r[0].x1,r[0].y1,r[0].x2-r[0].x1,r[0].y2-r[0].y1);
        cv::Mat faceImg = imgin(rect);
        bool flag = true;

        if(faceImg.rows> 40 &&faceImg.cols> 40 && r[0].label==1){
            clsprocess(faceImg,flag);
        }
        res=r[0];
        if(!flag)
            res.label=2;
    }
}

