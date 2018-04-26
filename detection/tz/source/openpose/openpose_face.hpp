#ifndef ALG_OPENPOSE_FACE_MX_A
#define ALG_OPENPOSE_FACE_MX_A
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>


class openpose_face
{
private:
    cv::Size net_input_sz_;
    std::string netname_;
    std::string modelname_;
    std::string outputname_;
    boost::shared_ptr<caffe::Net<float>> net_;
    boost::shared_ptr<caffe::Blob<float>> res_blob_;
    boost::shared_ptr<caffe::Blob<float>> heatmap_blob_;
    boost::shared_ptr<caffe::Blob<float>> peak_blob_;
    boost::shared_ptr<caffe::Blob<int>> kernel_blob_;


    int w_;
    int h_;
public:
    openpose_face(const std::string &modelpath, cv::Size netsz);
    void process(cv::Mat &img, std::vector<cv::Rect2f> &person_face, std::vector<std::vector<cv::Vec3f>> &pts);
    cv::Rect2f getfacefromposekeypoints(const std::vector<std::vector<cv::Vec3f>>& posekeypoints,
        unsigned int personindex, const unsigned int neck,
        const unsigned int headnose, const unsigned int lear, const unsigned int rear,
        const unsigned int leye, const unsigned int reye, const float threshold);

    float getDistance(cv::Vec3f pt1, cv::Vec3f pt2);

};

#endif
