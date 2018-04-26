#ifndef __RECOGNIZE_PLATE__H__
#define __RECOGNIZE_PLATE__H__
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include<iostream>
#include <vehicle_analysis/logodet.h>


using namespace caffe;
using  namespace std;

typedef struct _RESULT_T_
{
    cv::Rect rect;
    string plateNum;
    string color;
}RESULT_PLATE;


class Classifier
{
public:
    Classifier(const string& model_file, const string& trained_file, const string& mean_file);
    const float* Predict(const cv::Mat& img, const string& type);
    int RecognizePlate(cv::Mat& img, RESULT_PLATE& output);

private:
    void SetMean(const string& mean_file);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    bool use_mean_;
    cv::Mat mean_;
    std::vector<string> labels_;

public:
    bool bFirst;
};
#endif
