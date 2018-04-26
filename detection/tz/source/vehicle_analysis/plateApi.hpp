#ifndef __PLATE_API__H__
#define __PLATE_API__H__

#include "logodet.h"
#include "detectPlate.hpp"
#include "recognizePlate.hpp"


bool ScoreSort(vector<float> v1, vector<float> v2);

int DetectAndRecognizePlate(Detector& detector, Classifier& classifier, const cv::Mat& img, RESULT_PLATE& stResult);

#endif
