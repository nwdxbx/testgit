#ifndef ALG_OPENPOSE_MX_CONNECTOR_A
#define ALG_OPENPOSE_MX_CONNECTOR_A
#include <vector>
#include <opencv2/opencv.hpp>
#include "property.hpp"

template <typename T>
void connectBodyPartsCpu(std::vector< std::vector< cv::Vec<T, 3> > >& poseKeyPoints,
        const T* const heatMapPtr, const T* const peaksPtr,
        const property::PoseModel poseModel, const cv::Size& heatMapSize,
        const T scaleFactor
        );

#endif
