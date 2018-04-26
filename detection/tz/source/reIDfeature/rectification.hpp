#ifndef ALG_RECTIFICATION_A
#define ALG_RECTIFICATION_A
#include <opencv2/opencv.hpp>

template <typename T>
class rectification
{
public:
    static rectification<T> &ins();

    void process(cv::Mat &img, cv::Mat &rectifiedImg,
            cv::Rect rect, std::vector< cv::Point_<T> > pts);

protected:
    rectification();
    rectification(const rectification<T> &) = delete;
private:
    bool calcDirection(cv::Rect faceRect, std::vector<cv::Point_<T> > landmark);

private:
    std::vector< cv::Point_<T> > anchors_;
    cv::Size sz_;
};

template <typename T>
rectification<T> &rectification<T>::ins()
{
    static rectification obj;
    return obj;
}

template <typename T>
rectification<T>::rectification()
: sz_(cv::Size(96, 112))
{
    anchors_.push_back(cv::Point2f(30.2946, 51.6963));
    anchors_.push_back(cv::Point2f(65.5318, 51.5014));
    anchors_.push_back(cv::Point2f(48.0252, 71.7366));
    anchors_.push_back(cv::Point2f(33.5493, 92.3655));
    anchors_.push_back(cv::Point2f(62.7299, 92.2041));
}

template <typename T>
void rectification<T>::process(cv::Mat &img,
        cv::Mat &rectifiedImg, cv::Rect rect, std::vector<cv::Point_<T>> pts)
{
    //rectifiedImg = img;
    cv::Mat face = img(rect).clone();
    if (pts.size() != anchors_.size()){
        rectifiedImg = face;
        return;

    }

    if (!calcDirection(rect, pts))
    {
        rectifiedImg = face;
        return;
    }



    cv::Mat imgc=img.clone();
    //int h=img.rows;
    //int w=img.cols;
    //cv::resize(imgc, imgc, sz_);
    //for (auto &i : pts)
    //{
    //    i.x*=(float)sz_.width/w;
    //    i.y*=(float)sz_.height/h;
    //}


    cv::Mat warp_mat = cv::estimateRigidTransform(pts, anchors_, false);
    //cv::Mat warp_mat = cv::getAffineTransform(pts, anchors_);
    //cv::Mat warp_mat = cv::getPerspectiveTransform(pts, anchors_);
    //std::cout << warp_mat << std::endl;
    if (!warp_mat.empty())
    {
        cv::warpAffine(imgc, rectifiedImg, warp_mat, cv::Size(96, 112));
    }
    else
    {
        //rectifiedImg=cv::Mat();
        rectifiedImg = face;
    }
}

template <typename T>
bool rectification<T>::calcDirection(cv::Rect faceRect, std::vector<cv::Point_<T> > landmark)
{
    //frontal or profile
    float rect_len = std::sqrt(faceRect.width*faceRect.width*1.0/4
            +faceRect.height*faceRect.height*1.0/4);
    float nose_xdis = std::abs(landmark[2].x - (faceRect.x + faceRect.width/2));
    float nose_ydis = std::abs(landmark[2].y - (faceRect.y + faceRect.height/2));
    float face_len = std::sqrt( nose_xdis*nose_xdis + nose_ydis*nose_ydis);
    float dist = face_len/rect_len;
    float dist_threshold = 0.3;
    float nose_leye_dis = landmark[2].x - landmark[0].x;
    float nose_reye_dis = landmark[1].x - landmark[2].x;
    float nose_eys_dist = std::min(nose_leye_dis, nose_reye_dis)
        *1.0/std::max(nose_leye_dis, nose_reye_dis);
    float nose_eys_threshold = 0.3;
    if (dist < dist_threshold && nose_eys_dist > nose_eys_threshold)
        return true;
    return false;
}
#endif
