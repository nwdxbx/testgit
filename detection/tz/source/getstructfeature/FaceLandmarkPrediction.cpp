//
// Created by xulishuang on 18-2-24.
//

#include "FaceLandmarkPrediction.h"

using namespace std;
#define LAND_MARK_POINTS_COUNT 68

FaceLandmarkPrediction &FaceLandmarkPrediction:: ins() {
    static thread_local FaceLandmarkPrediction obj;
    return obj;
}

FaceLandmarkPrediction::FaceLandmarkPrediction() {
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);

    // init face_caffe_model
//    const string net_file = "./models/face_attrs_deploy_p";  // face_attrs_deploy.prototxt
//    net_.reset(new caffe::Net<float>(net_file, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));  //face_attrs.caffemodel
//    net_->CopyTrainedLayersFrom(
//            "./models/face_attrs_c");

//    const string net_file = "./models/face_attrs_deploy.prototxt";  // face_attrs_deploy.prototxt
//    net_.reset(new caffe::Net<float>(net_file, caffe::TEST));  //face_attrs.caffemodel
//    net_->CopyTrainedLayersFrom(
//            "./models/face_attrs.caffemodel");

    const string net_file = "./models/face_landmark_p";
    face_landmark_net_.reset(new caffe::Net<float>(net_file, caffe::TEST, "B743382C96DB85858FF65AE97DF98970", "802030C17462D3E3"));
    face_landmark_net_->CopyTrainedLayersFrom("./models/face_landmark_c");
}


void FaceLandmarkPrediction::predict(cv::Mat& face,cv::Rect& rt,PersonInfo &personInfo)
{
    cv::Mat face_roi = face(rt);
    auto input_layer_ = face_landmark_net_->input_blobs()[0];
    w_ = input_layer_->width();
    h_ = input_layer_->height();

    input_layer_->Reshape({1, input_layer_->shape(1),
                           input_layer_->shape(2), input_layer_->shape(3)});
    //cur_net_->Reshape();

    float *input_data = input_layer_->mutable_cpu_data();

    std::vector<std::vector<cv::Mat>> input_channels;
    input_channels.resize(input_layer_->shape()[0]);

    for (int i = 0; i < input_layer_->shape()[0]; ++i) {
        for (int j = 0; j < input_layer_->shape()[1]; ++j) {
            cv::Mat channel(input_layer_->height(), input_layer_->width(), CV_32FC1, input_data);
            input_channels[i].push_back(channel);

            input_data += input_layer_->width() * input_layer_->height();
        }

        cv::Mat x = face_roi.clone();
        if (x.empty()) {
            x = cv::Mat(w_, h_, CV_8UC3, cv::Scalar::all(0));
        } else {
            cv::resize(x, x, cv::Size(w_, h_));
        }

        if (x.channels() == 1) cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);

        cv::Mat floatImg, floatImg1;
        x.convertTo(floatImg1, CV_32FC3);

        floatImg = floatImg1.clone();

        cv::Scalar mean(127.5, 127.5, 127.5);
        floatImg -= mean; // minus mean value
        floatImg /= 128;

        cv::split(floatImg, input_channels[i]);
    }
    face_landmark_net_->Forward();

    // 是否口罩
    auto landmarks_blob  = face_landmark_net_->blob_by_name("conv6-4");
    auto landmarks_ptr = landmarks_blob->cpu_data();
    for(int i = 0; i < LAND_MARK_POINTS_COUNT; i++)
    {
        int pt_x = landmarks_ptr[i*2]*face_roi.cols+rt.x;
        int pt_y = landmarks_ptr[i*2+1]*face_roi.rows+rt.y;
        personInfo.face_landmarks.push_back(Vec2f(pt_x,pt_y));
    }
}