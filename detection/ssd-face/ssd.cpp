#include "ssd.hpp"
#include <glog/logging.h>

ssd_detect_caffe &ssd_detect_caffe::ins()
{
    static ssd_detect_caffe obj;
    return obj;
}

ssd_detect_caffe::ssd_detect_caffe()
    :netname_("./FPNdeploy.prototxt"),
      modelname_("./FPN_iter_38000.caffemodel")
{
    thresh_ = 0.6;
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    net_.reset(new caffe::Net<float>(netname_,caffe::TEST));
    net_->CopyTrainedLayersFrom(modelname_);
}

void ssd_detect_caffe::process(cv::Mat &imgin,ssd_detres &res)
{
    auto &input_layer = net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(127.5,127.5,127.5);

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
    sample =sample*0.0078125;
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
            int x1 = static_cast<int>(d[3] * dstimage.cols);
            int y1 = static_cast<int>(d[4] * dstimage.rows);
            int x2 = static_cast<int>(d[5] * dstimage.cols);
            int y2 = static_cast<int>(d[6] * dstimage.rows);
            det.x1 = std::max(x1,0);
            det.y1 = std::max(y1,0);
            det.x2 = std::min(x2,width-1);
            det.y2 = std::min(y2,height-1);

            det.score = score;
            r.push_back(det);
        }
    }
    std::sort(r.begin(),r.end(),[](const ssd_detres &lhs,const ssd_detres &rhs){return lhs.score>rhs.score;});
    if (r.size()==0)
        res=ssd_detres{-1,-1,-1,-1,-1};
    else
        res = r[0];
//    else{
//        if (r[0].y2 >= height)
//            res =ssd_detres{-1,-1,-1,-1,-1};
//        else
//            res =r[0];
//    }
}

