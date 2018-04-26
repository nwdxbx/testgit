//#include "getstructfeature/face_interface.h"
#include "caffe/caffe.hpp"
#include "../include/tuzhenalginterface.hpp"
#include "pedestrain_analysis/PedestrainAnalysis.h"
#include "vehicle_analysis/VehicleAnalysis.h"
#include "./mask_rcnn/MaskRCNNDetector.h"
#include "./crowd/CrowdEstimate.h"
#include "./openpose/openposeCaffe.hpp"
#include "version.h"


bool isKeypointsInBBox(std::vector<cv::Vec3f>& person_keypoints,cv::Rect rt){

    //bool is_keypoints_in_bbox = false;
    int points_inbbox_count = 0;
    for(cv::Vec3f& point:person_keypoints){
        if(rt.contains(cv::Point(point[0],point[1])) || point[2] <=0.01){
            points_inbbox_count+=1;
        }
    }
    float ratio = points_inbbox_count*1.0/person_keypoints.size();
    if(ratio >=0.85){
        return  true;
    }else{
        return false;
    }

}

cv::Rect getBBoxByKeyPoints(std::vector<cv::Vec3f>& person_keypoints){

    int x_min =10000, x_max = 0,y_max=0,y_min = 10000;
    for(cv::Vec3f& point:person_keypoints){
        if(point[2] > 0){
            if(point[0] < x_min){x_min = point[0];}
            if(point[0] > x_max){x_max = point[0];}
            if(point[1] < y_min){y_min = point[1];}
            if(point[1] > y_max){y_max = point[1];}
        }
    }
    return cv::Rect(x_min,y_min,x_max-x_min,y_max-y_min);
}

void get_version(std::string &str)
{
    str=VERSION_MAJOY+VERSION_MINER+VERSION_PATCH;
    //return ;
    //    LOG(INFO) << std::endl << std::endl;
    //    LOG(INFO) << "the alg version is: " << str << std::endl;
    //    LOG(INFO) << std::endl << std::endl;
}

void removeLowScore(std::vector<std::vector<cv::Vec3f> > &srcpts,std::vector<std::vector<cv::Vec3f> > &dstpts)
{
    dstpts.clear();
    for(size_t i=0;i<srcpts.size();i++)
    {
        if(((srcpts[i][9][2]>0.1) && (srcpts[i][10][2]>0.1)) || ((srcpts[i][12][2]>0.1) && (srcpts[i][13][2]>0.1)))
            dstpts.push_back(srcpts[i]);
    }
}

int remove_camera_by_id(void *pvTuzhenHandle,int cameraId)
{
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
    if(pTuzhenHandle)
    {
        std::map<int, PedestrainTracker*>::iterator iter = pTuzhenHandle->pedestrain_tracker.find(cameraId);
        if(iter != pTuzhenHandle->pedestrain_tracker.end())
        {
            delete pTuzhenHandle->pedestrain_tracker[cameraId];
            pTuzhenHandle->pedestrain_tracker.erase(iter);
        }

        std::map<int, VehicleTracker*>::iterator iter_vehicle = pTuzhenHandle->vehicle_tracker.find(cameraId);
        if(iter_vehicle != pTuzhenHandle->vehicle_tracker.end())
        {
            delete pTuzhenHandle->vehicle_tracker[cameraId];
            pTuzhenHandle->vehicle_tracker.erase(iter_vehicle);
        }
    }
    return EM_SUCESS_STATE;
}

int tuzhen_open(void **ppvTuzhenHandle, TuzhenPara *pTuzhenPara)
{
    TuzhenalgHandle *pTuzhenalgHandle;
    //pTuzhenalgHandle = (TuzhenalgHandle*)malloc(sizeof(TuzhenalgHandle));
    pTuzhenalgHandle = new TuzhenalgHandle();

    pTuzhenalgHandle->deviceID = pTuzhenPara->deviceID;
    pTuzhenalgHandle->eoptfun = pTuzhenPara->eoptfun;
    pTuzhenalgHandle->face_clarity_thresh = pTuzhenPara->face_clarity_thresh;
    pTuzhenalgHandle->face_min_width = pTuzhenPara->face_min_width;
    pTuzhenalgHandle->face_min_height = pTuzhenPara->face_min_height;
    pTuzhenalgHandle->mask_rcnn_prob_thresh = pTuzhenPara->mask_rcnn_prob_thresh;
    //pTuzhenalgHandle->trackline = 0;
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(pTuzhenalgHandle->deviceID);
    //init Net
    if(pTuzhenPara->eoptfun & EM_OPTIONAL_ITEM_CROWD_ESTIMATE){
        CrowdEstimate::ins();
    }
     if(pTuzhenPara->eoptfun & EM_OPTIONAL_ITEM_PEDESTRAIN)
    {
        MaskRCNNDetector::ins();
        PedestrainAnalysis::ins();
    }
    if(pTuzhenPara->eoptfun & EM_OPTIONAL_ITEM_VEHICLE)
    {
        MaskRCNNDetector::ins();
        VehicleAnalysis::ins();
    }
    *ppvTuzhenHandle = pTuzhenalgHandle;
    return 0;
}

int tuzhen_process(void *pvTuzhenHandle, TuzhenInput &tzInput, TuzhenOutput &tzOutput)
{
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
    if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_CROWD_ESTIMATE){
        CrowdEstimate::ins().process(tzInput.img,tzOutput.crowd_estimate_info.heat_map,tzOutput.crowd_estimate_info.estimated_pedestrain_count);
        //tzOutput.crowd_estimate_info.heat_map;
        return 0;
    }
    std::vector<caffe::Frcnn::BBox<float> > results;
    std::vector<cv::Mat>masks;
    std::vector<std::vector<cv::Vec3f>>keypoints;
    MaskRCNNDetector::ins().predict(tzInput.img,results,masks,keypoints,false,pTuzhenHandle->mask_rcnn_prob_thresh);
    std::vector<Pedestrain>pedestrains;
    std::vector<Vehicle>vehicles;
    for (size_t obj = 0; obj < results.size(); obj++) {
        int id =results[obj].id;
        cv::Rect rt = cv::Rect(results[obj][0],results[obj][1],int(results[obj][2]-results[obj][0]),int(results[obj][3]-results[obj][1]));
        if(id == DetectionType::PEDESTRAIN){
//            Pedestrain pedestrain;
//            //pedestrain.pedestrain_keypoints = keypoints[obj];
//            pedestrain.pedestrain_mask = masks[obj];
//            pedestrain.pedestrain_roi = rt;
//            pedestrain.pedestrain_score = results[obj].confidence;
//            pedestrains.push_back(pedestrain);
        }else if(id == DetectionType::CAR || id == DetectionType::BUS || id == DetectionType::TRUCK){
            Vehicle vehicle;
            vehicle.vehicle_roi = rt;
            vehicle.vehicle_mask = masks[obj];
            vehicle.vehicle_type = id;
            vehicle.vehicle_score = results[obj].confidence;
            vehicles.push_back(vehicle);
        }
    }
    //now the mask rcnn keypoints detect is not precise. use openpose instead.
    std::vector<std::vector<cv::Vec3f>> pts;
    pts.clear();
    openpose_caffe::ins().process(tzInput.img, pts);
//    pedestrains.clear();
    for(std::vector<cv::Vec3f>& person_keypoints : pts)
    {
//        for(Pedestrain& person: pedestrains){
//            if(isKeypointsInBBox(person_keypoints,person.pedestrain_roi)){
//                person.pedestrain_keypoints = person_keypoints;
//            }
//        }
        Pedestrain pedestrain;
        pedestrain.pedestrain_keypoints =person_keypoints;
        //pedestrain.pedestrain_mask = masks[obj];
        pedestrain.pedestrain_roi = getBBoxByKeyPoints(person_keypoints);
        pedestrain.pedestrain_score = 0.99;
        pedestrains.push_back(pedestrain);
    }
    //std::vector<tracker_obj_info>  tracker_output;
    if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_PEDESTRAIN)
    {
        PedestrainAnalysis::ins().predict_v3(pvTuzhenHandle,tzInput,pedestrains,tzOutput);
        //PedestrainAnalysis::ins().predict(pvTuzhenHandle,tzInput,tzOutput);
    }
    if(pTuzhenHandle->eoptfun & EM_OPTIONAL_ITEM_VEHICLE)
    {
        VehicleAnalysis::ins().predict(pvTuzhenHandle,tzInput,vehicles,tzOutput);
    }
    return 0;
}


int face_feature_attri(cv::Mat &img,PersonInfo& personinfo)//for face only;
{
    /*
     * get_head_infos function return value personinfo.bgetface maybe false while
     * Face_Re_ID::ins().process() detect a face,so there are something to correct.
    */
//    face_interface::ins().get_head_infos(img, personinfo);
//    std::vector<float> feature;
//    ///Face_Re_ID::ins().process(img, feature);
//    Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature);
//    //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
//    face_feat_entry entryface;
//    for (auto &f : feature)
//        entryface.add_reid_face_feat(f);
//    entryface.SerializeToString(&personinfo.facefea);
//    return EM_SUCESS_STATE;
    PedestrainAnalysis::ins().face_feature_attri(img,personinfo);
    return EM_SUCESS_STATE;
}

int feat_cal(cv::Mat &img, EFeature_Cal_Type flag, std::string&feature_str)
{

    PedestrainAnalysis::ins().feat_cal(img,flag,feature_str);
//    std::vector<float> feature;
//    switch(flag){
//        case EM_FEATURE_CAL_FACE:
//        {
//
//            Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature);
//           //Face_Re_ID::ins().process(img, feature);
//
//            //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
//            face_feat_entry entryface;
//            for (auto &f : feature)
//                entryface.add_reid_face_feat(f);
//            entryface.SerializeToString(&feature_str);
//
//            break;
//        }
//        case EM_FEATURE_CAL_PERSON:
//        {
//            Person_Re_ID::ins().process(img, feature);
//            human_feat_entry entryhuman;
//
//            //float to string size * 5 times
//            for (auto &f : feature)
//                entryhuman.add_reid_feat(f);
//            entryhuman.SerializeToString(&feature_str);
//            break;
//        }
//        default:
//            break;
//    }
//    if(feature.empty())
//        feature_str.clear();
    return EM_SUCESS_STATE;
}

int feature_cal_roi(cv::Mat &img, cv::Rect rt,EFeature_Cal_Type flag, std::string&res)//模型加载放到tuzhen_open同一个线程里边去进行;
{
    PedestrainAnalysis::ins().feature_cal_roi(img,rt,flag,res);
//    cv::Mat roi = img(rt);
//    std::vector<float> feature;
//    switch(flag){
//        case EM_FEATURE_CAL_FACE:
//        {
//
//            Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature);
//            //Face_Re_ID::ins().process(img, feature);
//
//            //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
//            face_feat_entry entryface;
//            for (auto &f : feature)
//                entryface.add_reid_face_feat(f);
//            entryface.SerializeToString(&res);
//
//            break;
//        }
//        case EM_FEATURE_CAL_PERSON:
//        {
//            Person_Re_ID::ins().process(img, feature);
//            human_feat_entry entryhuman;
//
//            //float to string size * 5 times
//            for (auto &f : feature)
//                entryhuman.add_reid_feat(f);
//            entryhuman.SerializeToString(&res);
//            break;
//        }
//        default:
//            break;
//    }
//    if(feature.empty())
//        res.clear();
    return EM_SUCESS_STATE;
}


int tuzhen_setpara(void *pvTuzhenHandle, TuzhenPara *pTuzhenPara)
{
    TuzhenalgHandle* pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
    pTuzhenHandle->eoptfun = pTuzhenPara->eoptfun;
    return 0;
}

void tuzhen_close(void *pvTuzhenHandle)
{
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
    if(pTuzhenHandle)
    {
        for(auto &item : pTuzhenHandle->pedestrain_tracker)
        {
            delete item.second;
        }
        for(auto &item : pTuzhenHandle->vehicle_tracker)
        {
            delete item.second;
        }
        //free(pTuzhenHandle);
        delete pTuzhenHandle;
        pTuzhenHandle = NULL;
    }
    return;
}
