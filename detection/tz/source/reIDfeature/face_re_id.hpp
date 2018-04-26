#ifndef _FACE_RE_ID_HPP_
#define _FACE_RE_ID_HPP_

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <functional>
#include "../../include/tuzhenalginterface.hpp"

class Face_Re_ID
{
public:
    static Face_Re_ID &ins();
    //void process(cv::Mat &img,cv::Mat &body, cv::Point offset_pt,std::vector<float> &result,float face_clarity_thresh);
    void process(cv::Mat &img,cv::Mat &body, cv::Point offset_pt,std::vector<float> &result,void *pvTuzhenHandle,PersonInfo &personInfo);
    void process(cv::Mat &img,cv::Mat &body, cv::Point offset_pt,std::vector<float> &result,PersonInfo &personInfo);
    void process(cv::Mat &face,std::vector<float> &result);
    void process(std::vector< cv::Mat> &imgin, std::vector<std::vector<float>> &result);
protected:
    Face_Re_ID(const std::string &net_file,
               const std::string &model_file);
    Face_Re_ID(const Face_Re_ID &)=delete;
    void preprocess(cv::Mat &img, cv::Mat &resimg);
    bool detectFace(cv::Mat& img, cv::Mat& body,cv::Point offset_pt, std::vector<cv::Mat>&faces_,void *pvTuzhenHandle,PersonInfo &personInfo);
    void get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res);
    int w_;
    int h_;

private:

    void rectify(cv::Mat &img,
                 std::vector<cv::Rect> &rect,
                 std::vector<std::vector<cv::Point>> &pt,
                 std::vector<cv::Mat> &rectified_img);
    const std::string netname_;
    const std::string modelname_;
    caffe::shared_ptr<caffe::Net<float>> net_;
    void clipBox(cv::Rect &rect, cv::Size imgsz);
    float modL2(std::vector<float>& feature);

};

#endif  //_FACE_RE_ID_HPP_
