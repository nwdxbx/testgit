#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "frcnn_param.hpp"
#include "frcnn_helper.hpp"

namespace FRCNN_API{

using std::vector;
using caffe::Blob;
using caffe::Net;
using caffe::Frcnn::FrcnnParam;
using caffe::Frcnn::Point4f;
using caffe::Frcnn::BBox;

    const int class_num=2;

class Detector {
public:
  Detector(std::string &proto_file, std::string &model_file,float thresh){
    Set_Model(proto_file, model_file);
      thresh_ = thresh;
  }
  void Set_Model(std::string &proto_file, std::string &model_file);
  void predict(const cv::Mat &img_in, vector<BBox<float> > &results,std::vector<cv::Mat>&masks,std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps,float prob_thresh);
  void predict_original(const cv::Mat &img_in, vector<BBox<float> > &results,std::vector<cv::Mat>&masks,std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps);
  void predict_iterative(const cv::Mat &img_in, vector<BBox<float> > &results,std::vector<cv::Mat>&masks,std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps);
    void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
    void boxes_sort(int num, const float* pred, float* sorted_pred);
private:
  void preprocess(const cv::Mat &img_in, const int blob_idx);
  void preprocess(const vector<float> &data, const int blob_idx);
  vector<boost::shared_ptr<Blob<float> > > predict(const vector<std::string> blob_names);
  boost::shared_ptr<Net<float> > net_;
  float mean_[3];
  int roi_pool_layer;
  float thresh_;
};

}
