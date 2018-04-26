//
// Created by xulishuang on 17-12-28.
//

#include "VehicleAnalysis.h"
#include "PlateDetector.h"
#include "car_type_color.h"
#include "../mask_rcnn/MaskRCNNDetector.h"
#include "../track/VehicleTracker.h"

void gclipBox2(cv::Rect &rect, cv::Size imgsz)
{
    cv::Rect k;
    k.x = std::min(std::max(0, rect.x), imgsz.width-1);
    k.y = std::min(std::max(0, rect.y), imgsz.height-1);
    k.width = std::min(std::max(0, rect.x+rect.width), imgsz.width)-k.x;
    k.height = std::min(std::max(0, rect.y+rect.height), imgsz.height)-k.y;

    rect=k;
}


VehicleAnalysis &VehicleAnalysis::ins()
{
    static thread_local VehicleAnalysis obj;
    return obj;
}

VehicleAnalysis::VehicleAnalysis() {

    car_type_color::ins();
    PlateDetector::ins();
}


void VehicleAnalysis::predict(void *pvTuzhenHandle,TuzhenInput &tzInput,std::vector<Vehicle>vehicles,TuzhenOutput &tzOutput)
{
    //std::vector<caffe::Frcnn::BBox<float> > results;
    //std::vector<cv::Mat>masks;
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle *) pvTuzhenHandle;

    if (tzInput.useROI == true) {
        tzInput.img = tzInput.img(tzInput.roi);
    }

    tzOutput.vehicles_info.clear();

    std::vector<det_input> rect_det;
    for (int i = 0; i < vehicles.size(); i++) {
        det_input obj_one;
        cv::Rect vehicle_rt = vehicles[i].vehicle_roi;
        obj_one.x = vehicle_rt.x;
        obj_one.y = vehicle_rt.y;
        obj_one.w = vehicle_rt.width;
        obj_one.h = vehicle_rt.height;
        obj_one.score = vehicles[i].vehicle_score;
        //obj_one.points = vehicles[i].pedestrain_keypoints;
        //obj_one.id = rect_out[i].id;
        rect_det.push_back(obj_one);
    }
    //pTuzhenHandle->pTracker[tzInput.cameraID]->process(tzInput.img, rect_det, tracker_output);
    std::vector<tracker_obj_info>  tracker_output;
    if(pTuzhenHandle->vehicle_tracker.find(tzInput.cameraID) == pTuzhenHandle->vehicle_tracker.end())
    {
        VehicleTracker *pTracker = new VehicleTracker();
        pTuzhenHandle->vehicle_tracker.insert(pair<int, VehicleTracker*>(tzInput.cameraID, pTracker));
    }
    pTuzhenHandle->vehicle_tracker[tzInput.cameraID]->process(tzInput.img, rect_det, tracker_output);
    if(!tracker_output.empty())
    {
        for(size_t i=0; i<tracker_output.size(); i++) {
            VehicleInfo vehicle_info;;
            tracker_obj_info &tracker_one = tracker_output[i];
            vehicle_info.vehicle_location = tracker_one.rect;
            //stPersonInfo.objectID = tracker_one.objectID;
            vehicle_info.breport = tracker_one.breport;
            vehicle_info.fscore = tracker_one.fcurscore;
            vehicle_info.objectID = tracker_one.objectID;
            //stPersonInfo.pts = tracker_one.points;
            if (vehicle_info.breport) {
                cv::Rect shrinkRect;
                shrinkRect.x = tracker_one.rect.x - tracker_one.rect.width / 4.0;
                shrinkRect.y = tracker_one.rect.y - tracker_one.rect.height / 6.0;
                shrinkRect.width = 3.0 / 2.0 * tracker_one.rect.width;
                shrinkRect.height = tracker_one.rect.height * 4.0 / 3.0;
                gclipBox2(shrinkRect, tzInput.img.size());

                Mat img_inflated = tzInput.img(shrinkRect);

                //MaskRCNNDetector::ins().predict(tzInput.img,results,masks);
                //cv::Rect rt = vehicles[i].vehicle_roi;
                cv::Mat vehicle_roi = tzInput.img(vehicle_info.vehicle_location);
                RESULT_PLATE result;
                vector<string> vehicle_brand_color;
                PlateDetector::ins().predict(img_inflated, result);
                car_type_color::ins().car_id(vehicle_roi, vehicle_brand_color);
                vehicle_info.card_no = result.plateNum;
                vehicle_info.card_color = result.color;
                vehicle_info.vehicle_brand = vehicle_brand_color[0];
                vehicle_info.vehicle_color = vehicle_brand_color[1];
            }
            tzOutput.vehicles_info.push_back(vehicle_info);
        }
    }
}