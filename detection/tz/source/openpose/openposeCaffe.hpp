#ifndef ALG_OPENPOSE_MX_A
#define ALG_OPENPOSE_MX_A
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include "property.hpp"

class openpose_caffe
{
public:
    using RES = std::vector< std::vector<cv::Vec3f> >;
    using BATCH_RES = std::vector< std::vector< std::vector<cv::Vec3f> > >;
    static openpose_caffe &ins();
    openpose_caffe();
    void process(const std::vector< cv::Mat > &imgs, 
            std::vector< std::vector< std::vector< cv::Vec3f> > > &res);

    void process(cv::Mat &img,
            std::vector< std::vector< cv::Vec3f> > &res);

    void getmatchpair(std::vector<std::pair<size_t, size_t> > &matchpair);

private:
    float resizeFixedAspectRatio(cv::Mat& cvMat,
            const cv::Size& targetSize, const int borderMode, const cv::Scalar& borderValue);

protected:

    openpose_caffe(const openpose_caffe &)=delete;
private:
    std::string netname_;
    std::string modelname_;
    std::string outputname_;
    property::PoseModel modelsel_;
    int numofparts_;
    int gpu_id_;
    boost::shared_ptr< caffe::Net<float> > net_;
    boost::shared_ptr< caffe::Blob<float> > res_blob_;
    boost::shared_ptr< caffe::Blob<float> > heatmap_blob_;
    boost::shared_ptr< caffe::Blob<float> > peak_blob_;
    boost::shared_ptr< caffe::Blob<int> > kernel_blob_;
    int w_;
    int h_;
};

#endif
