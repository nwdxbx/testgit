//
// Created by xulishuang on 17-12-28.
//

#ifndef PROJECT_PLATEDETECTOR_H
#define PROJECT_PLATEDETECTOR_H
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "detectPlate.hpp"
#include "recognizePlate.hpp"

class PlateDetector {
public:
    static PlateDetector &ins();
    void predict(const cv::Mat &img_in,RESULT_PLATE& result);

private:
    PlateDetector();
    boost::shared_ptr<Detector> plate_detector_;
    boost::shared_ptr<Classifier>plate_recognizer_;
};


#endif //PROJECT_PLATEDETECTOR_H
