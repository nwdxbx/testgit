#ifndef TUZHEN_ALG_FACE_REID_CAFFE_HPP
#define TUZHEN_ALG_FACE_REID_CAFFE_HPP
#include "simple_caffe.hpp"

namespace alg
{
class face_reid_caffe : public simple_caffe<std::vector<float>>
{
public:
    static face_reid_caffe &ins();

protected:
    face_reid_caffe();
    face_reid_caffe(const face_reid_caffe &)=delete;
    void preprocess(cv::Mat &img, cv::Mat &resimg);
    void get_res(const std::vector<caffe::Blob<float>*> &net_res, BATCH_RES &res);
};

}
#endif
