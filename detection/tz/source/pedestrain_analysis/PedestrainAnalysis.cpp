//
// Created by xulishuang on 17-12-28.
//
#include "../getstructfeature/face_interface.h"
#include "PedestrainAnalysis.h"
#include "../../include/tuzhenalginterface.hpp"
#include "../track/PedestrainTracker.h"
#include "../openpose/openposeCaffe.hpp"
//#include "../getstructfeature/getstructfeature.hpp"
//#include "../getstructfeature/humanAttribute.hpp"
#include "../reIDfeature/person_re_id.hpp"
#include "../reIDfeature/face_re_id.hpp"
#include "../crowd/crowd.h"
#include "compare/obj_reid_feat.prototxt.pb.h"
#include "../reIDfeature/person_re_id_v2.hpp"


void gclipBox(cv::Rect &rect, cv::Size imgsz)
{
    cv::Rect k;
    k.x = std::min(std::max(0, rect.x), imgsz.width-1);
    k.y = std::min(std::max(0, rect.y), imgsz.height-1);
    k.width = std::min(std::max(0, rect.x+rect.width), imgsz.width)-k.x;
    k.height = std::min(std::max(0, rect.y+rect.height), imgsz.height)-k.y;

    rect=k;
}



PedestrainAnalysis &PedestrainAnalysis::ins()
{
    static thread_local PedestrainAnalysis obj;
    return obj;
}

PedestrainAnalysis::PedestrainAnalysis() {
//    openpose_caffe::ins();
    //humanAttribute_caffe::ins();
    //face_interface::ins().loadYolo();
    face_interface::ins();
    Person_Re_ID_v2::ins();
//  Person_Re_ID::ins();
    Face_Re_ID::ins();
}


void PedestrainAnalysis::predict_v3(void *pvTuzhenHandle,TuzhenInput &tzInput, std::vector<Pedestrain>pedestrains,TuzhenOutput &tzOutput) {
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle *) pvTuzhenHandle;

    if (tzInput.useROI == true) {
        tzInput.img = tzInput.img(tzInput.roi);
    }

//uncomment when the keypoints is detected by mask rcnn
//    std::vector<std::vector<cv::Vec3f>> pts;
//    pts.clear();
//    for(int i = 0; i < pedestrains.size();i++){
//
//        cv::Rect person_rt = pedestrains[i].pedestrain_roi;
//        auto &pts_one=pedestrains[i].pedestrain_keypoints;
//        for (int i = 0; i < 17; i++) {
//            cv::Vec3f point = pts_one[i];
//            int x = point[0] * 1.0 / 28 * person_rt.width;
//            int y = point[1] * 1.0 / 28 * person_rt.height;
//            pts_one[i][0] = x +  person_rt.x;
//            pts_one[i][1] = y + person_rt.y;
//        }
//        cv::Vec3f point_neck;
//        point_neck[0] = (pts_one[5][0] + pts_one[6][0]) / 2;
//        point_neck[1] = (pts_one[5][1] + pts_one[6][1]) / 2;
//        point_neck[2] = (pts_one[5][2] + pts_one[6][2]) / 2;
//        pts_one.push_back(point_neck);
//        pts.push_back(pts_one);
//    }

    std::vector<tracker_obj_info>  tracker_output;

    std::vector<det_input> rect_det;
    for (int i = 0; i < pedestrains.size(); i++) {
        det_input obj_one;
        cv::Rect person_rt = pedestrains[i].pedestrain_roi;
        obj_one.x = person_rt.x;
        obj_one.y = person_rt.y;
        obj_one.w = person_rt.width;
        obj_one.h = person_rt.height;
        obj_one.score = pedestrains[i].pedestrain_score;
        obj_one.points = pedestrains[i].pedestrain_keypoints;
        //obj_one.id = rect_out[i].id;
        rect_det.push_back(obj_one);
    }
   //pTuzhenHandle->pTracker[tzInput.cameraID]->process(tzInput.img, rect_det, tracker_output);

    if(pTuzhenHandle->pedestrain_tracker.find(tzInput.cameraID) == pTuzhenHandle->pedestrain_tracker.end())
    {
        PedestrainTracker *pTracker = new PedestrainTracker();
        pTuzhenHandle->pedestrain_tracker.insert(pair<int, PedestrainTracker*>(tzInput.cameraID, pTracker));
    }
    pTuzhenHandle->pedestrain_tracker[tzInput.cameraID]->process(tzInput.img, rect_det, tracker_output);
    if(!tracker_output.empty())
    {
        for(size_t i=0; i<tracker_output.size(); i++) {
            PersonInfo stPersonInfo;
            tracker_obj_info &tracker_one = tracker_output[i];
            stPersonInfo.personrct = tracker_one.rect;
            stPersonInfo.objectID = tracker_one.objectID;
            stPersonInfo.breport = tracker_one.breport;
            stPersonInfo.fscore = tracker_one.fcurscore;
            stPersonInfo.pts = tracker_one.points;
            if (stPersonInfo.breport) {
                //for mask rcnn
//                cv::Rect shrinkRect;
//                shrinkRect.y = tracker_one.rect.y - tracker_one.rect.height/10.0;
//                shrinkRect.height = tracker_one.rect.height * 1.2;
//                shrinkRect.x = tracker_one.rect.x;
//                shrinkRect.width = tracker_one.rect.width;
//                gclipBox(shrinkRect, tzInput.img.size());

                //Mat img = tzInput.img(stPersonInfo.personrct);
                //if(shrinkRect.y < 0){shrinkRect.y = 0;}
                cv::Rect shrinkRect;
                shrinkRect.x = tracker_one.rect.x - tracker_one.rect.width/4.0;
                shrinkRect.y = tracker_one.rect.y - tracker_one.rect.height/6.0;
                shrinkRect.width = 3.0/2.0 * tracker_one.rect.width;
                shrinkRect.height = tracker_one.rect.height * 4.0/3.0;
                gclipBox(shrinkRect,tzInput.img.size());


                for(size_t i=0;i<tracker_one.points.size();i++)
                {
                    if(tracker_one.points[i][2]==0.)
                        continue;
                    tracker_one.points[i][0]=tracker_one.points[i][0]-shrinkRect.x;
                    tracker_one.points[i][1]=tracker_one.points[i][1]-shrinkRect.y;
                    if(tracker_one.points[i][0]<0) tracker_one.points[i][0]=0;
                    if(tracker_one.points[i][1]<0) tracker_one.points[i][1]=0;
                    if(tracker_one.points[i][0]>=shrinkRect.width) tracker_one.points[i][0]=shrinkRect.width-1;
                    if(tracker_one.points[i][1]>=shrinkRect.height) tracker_one.points[i][1]=shrinkRect.height-1;
                }

                Mat head_img = tzInput.img(shrinkRect);
                std::vector<float> facefeat;
                std::vector<float> personfeat;

                if(head_img.rows == 0  || head_img.cols == 0) continue;
                face_interface::ins().get_head_infos(head_img, stPersonInfo);
                if (stPersonInfo.bgetface && stPersonInfo.head_type == EM_PERSON_HEAD_FRONT) {
                    Mat headImg = head_img(stPersonInfo.facerct);
                    Face_Re_ID::ins().process(headImg, headImg, cv::Point(0, 0), facefeat,
                                              pTuzhenHandle,stPersonInfo);
                    if (facefeat.empty()) {
                        stPersonInfo.bgetface = false;
                    }
                }
                else
                {
                    stPersonInfo.bgetface = false;
                }

                //            Person_Re_ID::ins().process(img, personfeat);
                //            attributeAnalysis(img, pedestrains[i].pedestrain_keypoints, shrinkRect, stPersonInfo);

                            //check if leg points exist
                //if((stPersonInfo.pts[14][2]>0.1 && stPersonInfo.pts[16][2]>0.1) || (stPersonInfo.pts[13][2]>0.1 && stPersonInfo.pts[15][2]>0.1)) //mask rcnn
                if((stPersonInfo.pts[9][2]>0.1 && stPersonInfo.pts[10][2]>0.1) || (stPersonInfo.pts[12][2]>0.1 && stPersonInfo.pts[13][2]>0.1))
                {
                    int staPoint=0;
                    for(size_t i=0;i<stPersonInfo.pts.size();i++){
                        if(stPersonInfo.pts[i][2]>0.2)
                            staPoint++;
                    }
                    float fintegrity = 1.0*staPoint/stPersonInfo.pts.size();
                    if(fintegrity > tzInput.fintegritythresh)
                    {
                        stPersonInfo.person_integrity = fintegrity;
                        Person_Re_ID_det_res stperson;
                        Person_Re_ID_v2::ins().process(head_img, stperson);
                        personfeat = stperson.person_fea;
                        stPersonInfo.sleeve_length = static_cast<EPerson_Clothes_Lenght>(stperson.sleeveLength);
                        stPersonInfo.pants_length = static_cast<EPerson_Clothes_Lenght>(stperson.pantsLength);
                        stPersonInfo.sleeve_length_score = stperson.sleeveLengthScore;
                        stPersonInfo.pants_length_score = stperson.pantsLengthScore;
                        stPersonInfo.coat_color = static_cast<EPerson_Clothes_Color>(stperson.upclsColor);
                        stPersonInfo.trouser_color = static_cast<EPerson_Clothes_Color>(stperson.downclsColor);
                        stPersonInfo.coat_color_score = stperson.coat_color_score;
                        stPersonInfo.trouser_color_score = stperson.trouser_color_score;
//                        attributeAnalysis(img, pedestrains[i].pedestrain_keypoints, shrinkRect, stPersonInfo);
//                        Person_Re_ID::ins().process(img, personfeat);
                    }
                }

                if (facefeat.empty() && personfeat.empty()) {
                    stPersonInfo.breport = false;
                } else {
                    face_feat_entry entryface;
                    human_feat_entry entryhuman;

                    //float to string size * 5 times
                    for (auto &f : personfeat)
                        entryhuman.add_reid_feat(f);
                    entryhuman.SerializeToString(&stPersonInfo.personfea);

                    for (auto &f : facefeat)
                        entryface.add_reid_face_feat(f);
                    entryface.SerializeToString(&stPersonInfo.facefea);
                }

                stPersonInfo.personrct = shrinkRect;
#ifndef NDEBUG

                LOG(INFO) << "bare_score: " << stPersonInfo.bare_score << std::endl
                          << "fringe_score: " << stPersonInfo.fringe_score << std::endl
                          << "gender_score: " << stPersonInfo.gender_score << std::endl
                          << "glass_score: " << stPersonInfo.glass_score << std::endl
                          << "hat_score: " << stPersonInfo.hat_score << std::endl
                          << "mask_score: " << stPersonInfo.mask_score << std::endl
                          << "age_score: " << stPersonInfo.age_score << std::endl
                          << "head_score: " << stPersonInfo.head_score << std::endl
                          << "coat_color_score: " << stPersonInfo.coat_color_score << std::endl
                          << "trouser_color_score: " << stPersonInfo.trouser_color_score << std::endl
                          << "sleeve_length_score: " << stPersonInfo.sleeve_length_score << std::endl
                          << "pants_length_score: " << stPersonInfo.pants_length_score << std::endl
                          << "person_integrity: " << stPersonInfo.person_integrity << std::endl;
#endif
            }
            tzOutput.personvec.push_back(stPersonInfo);
        }
    }
}


void PedestrainAnalysis::predict(void *pvTuzhenHandle,TuzhenInput &tzInput, TuzhenOutput &tzOutput)
{
//    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle*)pvTuzhenHandle;
//
//    if(tzInput.useROI == true){
//        tzInput.img = tzInput.img(tzInput.roi);
//    }
//
//    std::vector<std::vector<cv::Vec3f>> pts;
//    pts.clear();
//    openpose_caffe::ins().process(tzInput.img, pts);
//    std::vector<tracker_obj_info>  tracker_output;
//
//
//    if(pTuzhenHandle->pedestrain_tracker.find(tzInput.cameraID) == pTuzhenHandle->pedestrain_tracker.end())
//    {
//        PedestrainTracker *pTracker = new PedestrainTracker();
//        pTuzhenHandle->pedestrain_tracker.insert(pair<int, PedestrainTracker*>(tzInput.cameraID, pTracker));
//    }
//    //pTuzhenHandle->pedestrain_tracker[tzInput.cameraID]->process(tzInput.img, pts, tracker_output);
//    if(!tracker_output.empty())
//    {
//        for(size_t i=0; i<tracker_output.size(); i++)
//        {
//            PersonInfo stPersonInfo;
//            tracker_obj_info &tracker_one = tracker_output[i];
//
//            stPersonInfo.personrct = tracker_one.rect;
//            stPersonInfo.objectID = tracker_one.objectID;
//            stPersonInfo.breport = tracker_one.breport;
//            stPersonInfo.fscore = tracker_one.fcurscore;
//            stPersonInfo.pts = tracker_one.points;
//
//            if(tracker_one.breport)
//            {
//                cv::Rect shrinkRect;
//                shrinkRect.x = tracker_one.rect.x - tracker_one.rect.width/4.0;
//                shrinkRect.y = tracker_one.rect.y - tracker_one.rect.height/6.0;
//                shrinkRect.width = 3.0/2.0 * tracker_one.rect.width;
//                shrinkRect.height = tracker_one.rect.height * 4.0/3.0;
//                gclipBox(shrinkRect,tzInput.img.size());
//
//
//                for(size_t i=0;i<tracker_one.points.size();i++)
//                {
//                    if(tracker_one.points[i][2]==0.)
//                        continue;
//                    tracker_one.points[i][0]=tracker_one.points[i][0]-shrinkRect.x;
//                    tracker_one.points[i][1]=tracker_one.points[i][1]-shrinkRect.y;
//                    if(tracker_one.points[i][0]<0) tracker_one.points[i][0]=0;
//                    if(tracker_one.points[i][1]<0) tracker_one.points[i][1]=0;
//                    if(tracker_one.points[i][0]>=shrinkRect.width) tracker_one.points[i][0]=shrinkRect.width-1;
//                    if(tracker_one.points[i][1]>=shrinkRect.height) tracker_one.points[i][1]=shrinkRect.height-1;
//                }
//
//                Mat img = tzInput.img(shrinkRect);
//
//
//                std::vector<float> facefeat;
//                std::vector<float> personfeat;
//
//                face_interface::ins().get_head_infos(img, stPersonInfo);
//                if(stPersonInfo.bgetface)
//                {
//                    Mat headImg = img(stPersonInfo.facerct);
//                    Face_Re_ID::ins().process(headImg,headImg,cv::Point(0,0),facefeat);
//                    if(facefeat.empty())
//                    {
//                        stPersonInfo.bgetface = false;
//                    }
//                }
//
//                //check if leg points exist
//                if((stPersonInfo.pts[9][2]>0.1 && stPersonInfo.pts[10][2]>0.1) || (stPersonInfo.pts[12][2]>0.1 && stPersonInfo.pts[13][2]>0.1))
//                {
//                    int staPoint=0;
//                    for(size_t i=0;i<stPersonInfo.pts.size();i++){
//                        if(stPersonInfo.pts[i][2]>0.2)
//                            staPoint++;
//                    }
//                    float fintegrity = 1.0*staPoint/stPersonInfo.pts.size();
//                    if(fintegrity > tzInput.fintegritythresh)
//                    {
//                        stPersonInfo.person_integrity = fintegrity;
//
//                        attributeAnalysis(img, tracker_one.points, shrinkRect, stPersonInfo);
//                        //Person_Re_ID_v2::ins().process(img, personfeat);
//                        Person_Re_ID_det_res stperson;
//                        Person_Re_ID_v2::ins().process(img, stperson);
//                        personfeat = stperson.person_fea;
//                    }
//                }
//
//                if(facefeat.empty() && personfeat.empty())
//                {
//                    stPersonInfo.breport = false;
//                }
//                else
//                {
//                    face_feat_entry entryface;
//                    human_feat_entry entryhuman;
//
//                    //float to string size * 5 times
//                    for (auto &f : personfeat)
//                        entryhuman.add_reid_feat(f);
//                    entryhuman.SerializeToString(&stPersonInfo.personfea);
//
//                    for (auto &f : facefeat)
//                        entryface.add_reid_face_feat(f);
//                    entryface.SerializeToString(&stPersonInfo.facefea);
//                }
//
//                //for debug
//                stPersonInfo.personrct = shrinkRect;
//
//#ifndef NDEBUG
//
//                LOG(INFO) << "bare_score: " << stPersonInfo.bare_score << std::endl
//                          << "fringe_score: " << stPersonInfo.fringe_score << std::endl
//                          << "gender_score: " << stPersonInfo.gender_score << std::endl
//                          << "glass_score: " << stPersonInfo.glass_score << std::endl
//                          << "hat_score: " << stPersonInfo.hat_score << std::endl
//                          << "mask_score: " << stPersonInfo.mask_score << std::endl
//                          << "age_score: " << stPersonInfo.age_score << std::endl
//                          << "head_score: " << stPersonInfo.head_score << std::endl
//                          << "coat_color_score: " << stPersonInfo.coat_color_score << std::endl
//                          << "trouser_color_score: " << stPersonInfo.trouser_color_score << std::endl
//                          << "sleeve_length_score: " << stPersonInfo.sleeve_length_score << std::endl
//                          << "pants_length_score: " << stPersonInfo.pants_length_score << std::endl
//                          << "person_integrity: " << stPersonInfo.person_integrity << std::endl;
//#endif
//
//            }
//            tzOutput.personvec.push_back(stPersonInfo);
//
//        }
//    }
}


int PedestrainAnalysis::face_feature_attri(cv::Mat &img,PersonInfo& personinfo)//for face only;
{
    /*
     * get_head_infos function return value personinfo.bgetface maybe false while
     * Face_Re_ID::ins().process() detect a face,so there are something to correct.
    */
    face_interface::ins().get_head_infos(img, personinfo);
    std::vector<float> feature;
    ///Face_Re_ID::ins().process(img, feature);
    Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature,personinfo);
    //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
    face_feat_entry entryface;
    for (auto &f : feature)
        entryface.add_reid_face_feat(f);
    entryface.SerializeToString(&personinfo.facefea);
    return EM_SUCESS_STATE;
}

int PedestrainAnalysis::feat_cal(cv::Mat &img, EFeature_Cal_Type flag, std::string&feature_str)
{

    std::vector<float> feature;
    switch(flag){
        case EM_FEATURE_CAL_FACE:
        {
            PersonInfo personInfo;

            Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature,personInfo);
            //Face_Re_ID::ins().process(img, feature);

            //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
            face_feat_entry entryface;
            for (auto &f : feature)
                entryface.add_reid_face_feat(f);
            entryface.SerializeToString(&feature_str);

            break;
        }
        case EM_FEATURE_CAL_PERSON:
        {
            //Person_Re_ID::ins().process(img, feature);
            Person_Re_ID_det_res stperson;
            Person_Re_ID_v2::ins().process(img, stperson);
            feature = stperson.person_fea;
            human_feat_entry entryhuman;

            //float to string size * 5 times
            for (auto &f : feature)
                entryhuman.add_reid_feat(f);
            entryhuman.SerializeToString(&feature_str);
            break;
        }
        default:
            break;
    }
    if(feature.empty())
        feature_str.clear();
    return EM_SUCESS_STATE;
}

int PedestrainAnalysis::feature_cal_roi(cv::Mat &img, cv::Rect rt,EFeature_Cal_Type flag, std::string&res)//模型加载放到tuzhen_open同一个线程里边去进行;
{
    cv::Mat roi = img(rt);
    std::vector<float> feature;
    switch(flag){
        case EM_FEATURE_CAL_FACE:
        {

            PersonInfo personInfo;
            Face_Re_ID::ins().process(img,img,cv::Point(0,0),feature,personInfo);
            //Face_Re_ID::ins().process(img, feature);

            //CHECK_EQ(feature.size(), 512) << "feat size not right -------------------------!";
            face_feat_entry entryface;
            for (auto &f : feature)
                entryface.add_reid_face_feat(f);
            entryface.SerializeToString(&res);

            break;
        }
        case EM_FEATURE_CAL_PERSON:
        {
            //Person_Re_ID::ins().process(img, feature);
            Person_Re_ID_det_res stperson;
            Person_Re_ID_v2::ins().process(img, stperson);
            feature = stperson.person_fea;
            human_feat_entry entryhuman;

            //float to string size * 5 times
            for (auto &f : feature)
                entryhuman.add_reid_feat(f);
            entryhuman.SerializeToString(&res);
            break;
        }
        default:
            break;
    }
    if(feature.empty())
        res.clear();
    return EM_SUCESS_STATE;
}
