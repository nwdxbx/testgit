//
// Created by xulishuang on 17-12-28.
//

#ifndef PROJECT_VEHICLEDETECTOR_H
#define PROJECT_VEHICLEDETECTOR_H
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../../include/tuzhenalginterface.hpp"


typedef struct Vehicle {
    Rect vehicle_roi;      //行人目标框
    cv::Mat vehicle_mask;
    int vehicle_type;
    float vehicle_score;
};

class VehicleAnalysis {
public:
    static VehicleAnalysis &ins();
    void predict(void *pvTuzhenHandle,TuzhenInput &tzInput,std::vector<Vehicle>vehicles,TuzhenOutput &tzOutput);

private:
    VehicleAnalysis();
};


#endif //PROJECT_VEHICLEDETECTOR_H
