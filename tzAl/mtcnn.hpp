#ifndef TUZHEN_ALG_MTCNN_HPP
#define TUZHEN_ALG_MTCNN_HPP
// caffe
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layer_factory.hpp>
// c++
#include <string>
#include <vector>
#include <fstream>
// opencv
#include <opencv2/opencv.hpp>


namespace alg
{
class MTCnn
{
public:
    int numFace;

public:
    struct FaceInf
    {
        cv::Rect faceRect;
        cv::Point leye;
        cv::Point reye;
        cv::Point nose;
        cv::Point lmouth;
        cv::Point rmouth;
        float score;
        bool isCorrect;
    };
    using BATCH_RES=std::vector< std::vector<FaceInf> >;
    using RES=std::vector< std::vector<FaceInf> >;

    static MTCnn &ins();

    void Detect(cv::Mat &images, std::vector<FaceInf> &result_Inf);
    void Detect(std::vector<cv::Mat> &images, std::vector<std::vector<FaceInf>> &result_Inf);

protected:
    MTCnn();
    MTCnn(const MTCnn &)=delete;
private:
    typedef struct FaceRect
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
    } FaceRect;
    typedef struct FacePts
    {
        float x[5], y[5];
    } FacePts;
    typedef struct FaceInfo
    {
        FaceRect bbox;
        cv::Vec4f regression;
        FacePts facePts;
        float roll;
        float pitch;
        float yaw;
    } FaceInfo;

    bool CompareBBox(std::vector<FaceInfo>& a);
    bool CompareBBox(std::vector<FaceInfo>& a, std::vector<bool>& b);
    void GenerateBoundingBox(caffe::Blob<float>* confidence, caffe::Blob<float>* reg, float scale, float thresh, int img_idx);
    void ClassifyFace_1(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single, std::vector<float> &score);
    void ClassifyFace_2(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single, std::vector<float> &score, std::vector<bool> &frontalLabel);
    std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    std::vector<FaceInfo> NonMaximumSuppression(std::vector<MTCnn::FaceInfo>& bboxes, float thresh,std::vector<bool> srcFrontal, std::vector<bool>& dstFrontal);
    void Bbox2Square(std::vector<FaceInfo>& bboxes);
    void Padding(int img_w, int img_h,std::vector<FaceInfo> &regressed_rects_item_i);
    std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);

private:
    float threshold[3];
    float factor = 0.709;
    int minSize = 20;

    boost::shared_ptr<caffe::Net<float> > PNet_;
    boost::shared_ptr<caffe::Net<float> > RNet_;
    boost::shared_ptr<caffe::Net<float> > ONet_;

    caffe::Blob<float>* Rnet_crop_input_layer;
    caffe::Blob<float>* Onet_crop_input_layer;

    float* Rnet_input_data;
    float* Onet_input_data;

    std::vector<FaceInfo> regressed_pading_;
    std::vector<FaceInfo> condidate_rects_;
};
}

#endif
