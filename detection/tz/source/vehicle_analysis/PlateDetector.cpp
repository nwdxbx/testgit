//
// Created by xulishuang on 17-12-28.
//

#include "PlateDetector.h"
#include "plateApi.hpp"
#include <gflags/gflags.h>


DEFINE_string(mean_file, "",
"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
"If specified, can be one value or can be same as image channels"
" - would subtract from the corresponding channel). Separated by ','."
"Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
"The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
"If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.16,
"Only store detections with score higher than the threshold.");

using  namespace std;


PlateDetector &PlateDetector::ins()
{
    static thread_local PlateDetector obj;
    return obj;
}

PlateDetector::PlateDetector()
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(gpuid);
    //net_.reset(new caffe::Net<float>(net_file, caffe::TEST));
    //net_->CopyTrainedLayersFrom(model_file);

    const string& model_file = "./models/ss_chebiao_p";
    const string& weights_file = "./models/ss_chebiao_c";
    const string& model_file_pcr   = "./models/mylstm_plate_p";
    const string& weights_file_ocr = "./models/mylstm_iter_plate_c";
    const string& mean_file_ocr = "";

//    const string& model_file = "./models/plate/ss_chebiao_deploy.prototxt";
//    const string& weights_file = "./models/plate/ss_chebiao.caffemodel";
//    const string& model_file_pcr   = "./models/plate/plate_deploy.prototxt";
//    const string& weights_file_ocr = "./models/plate/mylstm_iter_3100.caffemodel";
//    const string& mean_file_ocr = "";

    //std::ifstream infile("/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/ssd/list.txt");

    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    plate_detector_.reset(new Detector(model_file, weights_file, mean_file, mean_value));
    plate_recognizer_.reset(new Classifier(model_file_pcr, weights_file_ocr, mean_file_ocr));
}

void PlateDetector::predict(const cv::Mat &img_in,RESULT_PLATE& result)
{
    int iRet = DetectAndRecognizePlate(*plate_detector_, *plate_recognizer_, img_in, result);
    if (iRet == -1)
    {
        cout << "[ERROR]: DetectAndRecognizePlate faild!!" << endl;
    }
}
