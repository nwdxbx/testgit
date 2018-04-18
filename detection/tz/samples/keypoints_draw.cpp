//
// Created by xulishuang on 17-12-20.
//
#include <opencv2/opencv.hpp>
#include "../source/openpose/openposeCaffe.hpp"
#include <string>



int main(void) {

    std::vector<cv::Scalar> colors_;

    colors_.push_back(cv::Scalar(0, 255, 255));
    colors_.push_back(cv::Scalar(0, 255, 0));
    colors_.push_back(cv::Scalar(255, 255, 0));
    colors_.push_back(cv::Scalar(255, 0, 255));
    colors_.push_back(cv::Scalar(0, 0, 255));
    colors_.push_back(cv::Scalar(255, 125, 0));
    colors_.push_back(cv::Scalar(125, 255, 0));
    colors_.push_back(cv::Scalar(0, 255, 125));
    colors_.push_back(cv::Scalar(0, 125, 255));
    colors_.push_back(cv::Scalar(255, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 255));
    colors_.push_back(cv::Scalar(125, 125, 255));
    colors_.push_back(cv::Scalar(125, 255, 125));
    colors_.push_back(cv::Scalar(255, 125, 125));
    colors_.push_back(cv::Scalar(0, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 0));
    colors_.push_back(cv::Scalar(0, 125, 0));


    std::string file = "/work/dev/experiments/py-mask-rcnn/data/demo/000011_000368_00001000.jpg";
    cv::Mat frame = cv::imread(file);
    std::vector<std::vector<cv::Vec3f>> pts;
    pts.clear();
    openpose_caffe::ins().process(frame, pts);

    std::vector<std::pair<size_t, size_t>> matchpair;
    //openpose_caffe::ins().getmatchpair(matchpair);
    const std::vector<unsigned int> render_pairs {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17};
    //std::vector<std::pair<size_t, size_t> > matchpair;
    for (size_t i=0; i<render_pairs.size()/2; i++)
        matchpair.push_back({render_pairs[2*i], render_pairs[2*i+1]});

    std::cout << pts.size();
    for (size_t i=0; i<pts.size(); i++)
    {
        auto &pts_one=pts[i];
        if (pts_one.empty())
            continue;

        auto f=[&](int idpart, const cv::Scalar &color)
        {
            std::pair<size_t, size_t> &bodypart=matchpair[idpart];
            if (std::min(pts_one[bodypart.first][2], pts_one[bodypart.second][2]) < 0.1)
                return;
            int id1=bodypart.first;
            int id2=bodypart.second;
            cv::line(frame,
                     cv::Point(pts_one[id1][0], pts_one[id1][1]),
                     cv::Point(pts_one[id2][0], pts_one[id2][1]),
                     color, 2);
        };
        for (size_t idpart=0; idpart<matchpair.size(); idpart++)
        {
            f(idpart, colors_[idpart]);
        }
    }

    cv::imshow("find-image", frame);
    cv::waitKey(0);
}
