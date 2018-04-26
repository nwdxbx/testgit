//
// Created by xulishuang on 18-1-22.
//

#include "PedestrainTracker.h"
#include "trackNet.h"
#include "graph_min_cut_assign.hpp"


void PedestrainTracker::process(cv::Mat srcImg, std::vector< det_input> &persons, std::vector<tracker_obj_info> &tracker_output)
{
    //LOG(INFO) << "start tracking --------------------------------------------------" <<std::endl << std::endl;
//    const int interval = 10;
//    const int static_interval = 30;

//    if(objrects.size()==0)
//    {
//        return;
//    }

//    for(int i=objrects.size()-1;i>-1;i--)
//    {
//        if(objrects[i].w>srcImg.cols/3)
//        {
//            objrects.erase(objrects.begin()+i);
//        }

//    }

//    std::vector<double> rectScore;
//    getRectScore(objrects,rectScore);


//    //step1: compute SM Matrix, M X N, M: observe dim; N: tracker dim
//    std::vector< cv::Mat> imgin;
//    for(int i=0;i<objrects.size();i++)
//    {
//        imgin.push_back(srcImg(cv::Rect(objrects[i].x,objrects[i].y,objrects[i].w,objrects[i].h)));
//    }

//    std::vector<std::vector<float>> objFeature;

////    double start = cv::getTickCount();
//    //process2(imgin,objFeature);
//    trackNet::ins().process(imgin,objFeature);
////    double end = cv::getTickCount();
////    LOG(INFO) << "track cal time: [ " << (end-start)*1000/imgin.size()/cv::getTickFrequency() << "ms ]";
////    LOG(INFO) << std::endl;
////    LOG(INFO) << "track cal time: [ " << (end-start)*1000/cv::getTickFrequency() << "ms ]";
////    LOG(INFO) << std::endl;

//    cv::Mat SM=cv::Mat(objFeature.size(), trackers_.size(), CV_32F);
//    similarityMatrix(objFeature,SM);
//    cv::normalize(SM,SM,1.0,0.0,cv::NORM_MINMAX);

//    cv::Mat DM=cv::Mat(objrects.size(), trackers_.size(), CV_32F);
//    double distThresh = 300;
//    distanceMatrix(objrects,DM,distThresh);
//    cv::normalize(DM,DM,1.0,0.0,cv::NORM_MINMAX);
//    DM=1-DM;

//    cv::multiply(SM,DM,SM);

//    //step2: graphMinCut to get if the observe rect matched or not, assign dim: M
//    //if matched, assign value is the tracker index, else -1
//    std::vector<int> assign;
//    if(!SM.empty())
//    {
//        SM=1-SM;
//        GraphAssign::graphMinCut(SM, assign);
//        //std::cout << SM << std::endl;

//    }
//    else
//    {
//        for(auto &i : objrects)
//        {
//            assign.push_back(-1);
//        }

//    }

//    //step3: add new tracker
//    //judge the track matched observe rects or not
//    std::vector<bool> istracked;
//    for(auto &i : trackers_)
//    {
//        istracked.push_back(false);
//    }

//    for(size_t i=0; i<assign.size(); i++)
//    {
//        //not match, add new tracker
//        if(assign[i] == -1)
//        {
//            tracker_obj_info info_one;

//            info_one.rect= cv::Rect(objrects[i].x,objrects[i].y,objrects[i].w,objrects[i].h);
//            info_one.points = objrects[i].points;
//            info_one.fcurscore = rectScore[i]*objrects[i].score;
////            info_one.fcurscore = rectScore[i];
//            info_one.objectID = objectIDLast + 1;
//            info_one.trackerlife = 1;

//            //deep learning for object feature
//            info_one.lastOneFeature = objFeature[i];

//            if(info_one.fcurscore > object_tracker::fminreportscore)
//            {
//                info_one.breport = true;
//                info_one.fmaxscore = info_one.fcurscore;
//            }
//            else
//            {
//                info_one.breport = false;
//                info_one.fmaxscore = -1.0f;
//            }
//            objectIDLast++;
//            trackers_.push_back(info_one);
//            istracked.push_back(true);
//        }
//        else
//        {
//            tracker_obj_info &tracker_match = trackers_[assign[i]];
//            //update tracker
//            tracker_match.rect = cv::Rect(objrects[i].x,objrects[i].y,objrects[i].w,objrects[i].h);
//            //deep learning for object feature
//            tracker_match.lastOneFeature = objFeature[i];
//            tracker_match.points = objrects[i].points;

//            tracker_match.fcurscore = rectScore[i]*objrects[i].score;
////            tracker_match.fcurscore = rectScore[i];
//            if(tracker_match.fcurscore > tracker_match.fmaxscore && tracker_match.fcurscore > object_tracker::fminreportscore)
//                tracker_match.breport = true;
//            else
//                tracker_match.breport = false;

//            if(tracker_match.fmaxscore < tracker_match.fcurscore)
//                tracker_match.fmaxscore = tracker_match.fcurscore;


//            tracker_match.trackerlife++;
//            istracked[assign[i]] = true;
//        }
//    }

//    //step4: delete tracker that not match any observe rect
//    //CHECK_EQ(istracked.size(), trackers_.size());

//    for(size_t i = 0; i<trackers_.size();)
//    {
//        if(!istracked[i])
//        {
//            istracked.erase(istracked.begin() + i);
//            trackers_.erase(trackers_.begin() + i);
//        }
//        else
//            i++;
//    }
//    tracker_output = trackers_;
//    return;

    const int interval = 10;
    const int static_interval = 30;

    //step0: transfer the points to rects
    std::vector<cv::Rect> objrects;
    std::vector<vector<cv::Vec3f> > newpoints;



//    if(!points.empty())
//    {
//        for(size_t i=0; i<points.size(); i++)
//        {
//            cv::Point lt(10000,10000), rb(-10000,-10000);
//            for (size_t j=0; j<points[i].size(); j++)
//            {
//                if (points[i][j][2]>0.2f)
//                {
//                    lt=cv::Point(std::min(lt.x, (int)points[i][j][0]), std::min(lt.y, (int)points[i][j][1]));
//                    rb=cv::Point(std::max(rb.x, (int)points[i][j][0]), std::max(rb.y, (int)points[i][j][1]));
//                }
//            }
//            //only accept the object rect bigger than 10x15
//            if(rb.x - lt.x > 10 && rb.y - lt.y > 15)
//            {
//                objrects.push_back(cv::Rect(lt, rb));
//                newpoints.push_back(points[i]);
//            }
//        }
//    }

    for (int i = 0; i < persons.size();i++){
        objrects.push_back(cv::Rect(persons[i].x,persons[i].y,persons[i].w,persons[i].h));
    }
    std::vector<double> rectScore;
    getRectScore(persons,rectScore);

    //compute observe rect hists
    std::vector<cv::Mat> vecMask, vecHist;
    genBodyMask(srcImg, objrects, vecMask);
    genHist(srcImg, vecMask, vecHist);

    //step1: compute SM Matrix, M X N, M: observe dim; N: tracker dim

    cv::Mat SM=cv::Mat(objrects.size(), trackers_.size(), CV_32F);
    similarityMatrix(vecHist,SM);

    cv::Mat DM=cv::Mat(objrects.size(), trackers_.size(), CV_32F);
    distanceMatrix(objrects,DM);
    cv::normalize(DM,DM,1.0,0.0,cv::NORM_MINMAX);
    DM=1-DM;

    cv::multiply(SM,DM,SM);

    //step2: graphMinCut to get if the observe rect matched or not, assign dim: M
    //if matched, assign value is the tracker index, else -1
    std::vector<int> assign;
    if(!SM.empty())
    {
        SM=1-SM;
        GraphAssign::graphMinCut(SM, assign);
        //std::cout << SM << std::endl;

    }
    else
    {
        for(auto &i : objrects)
        {
            assign.push_back(-1);
        }

    }

    //step3: add new tracker
    //judge the track matched observe rects or not
    std::vector<bool> istracked;
    for(auto &i : trackers_)
    {
        istracked.push_back(false);
    }

    for(size_t i=0; i<assign.size(); i++)
    {
        //not match, add new tracker
        if(assign[i] == -1)
        {
            tracker_obj_info info_one;

            //info_one.fcurscore = getScore(points[i], objrects[i]);
            info_one.fcurscore = rectScore[i]*persons[i].score;
            info_one.lastHist = vecHist[i];
            info_one.objectID = objectIDLast + 1;
            info_one.trackerlife = 1;
            info_one.rect = objrects[i];
            info_one.points = persons[i].points;
            //info_one.points = newpoints[i];
            if(info_one.fcurscore > object_tracker::fminreportscore)
            {
                info_one.breport = true;
                info_one.fmaxscore = info_one.fcurscore;
            }
            else
            {
                info_one.breport = false;
                info_one.fmaxscore = -1.0f;
            }
            objectIDLast++;
            trackers_.push_back(info_one);
            istracked.push_back(true);
        }
        else
        {
            tracker_obj_info &tracker_match = trackers_[assign[i]];
            //update tracker
            tracker_match.rect = objrects[i];
            //tracker_match.points = newpoints[i];
            tracker_match.points = persons[i].points;
            tracker_match.lastHist = vecHist[i];
            //tracker_match.fcurscore = getScore(points[i],objrects[i]);
            tracker_match.fcurscore = rectScore[i]*persons[i].score;
            if(tracker_match.fcurscore > tracker_match.fmaxscore && tracker_match.fcurscore > object_tracker::fminreportscore)
                tracker_match.breport = true;
            //temp for show
            //if((tracker_match.fcurscore > tracker_match.fmaxscore || (tracker_match.fcurscore > tracker_match.fmaxscore*0.9 && tracker_match.trackerlife % 5 == 0)) && tracker_match.fcurscore > object_tracker::fminreportscore)
                //tracker_match.breport = true;
            else
                tracker_match.breport = false;

            if(tracker_match.fmaxscore < tracker_match.fcurscore)
                tracker_match.fmaxscore = tracker_match.fcurscore;


            tracker_match.trackerlife++;
            istracked[assign[i]] = true;
        }
    }

    //step4: delete tracker that not match any observe rect
    //CHECK_EQ(istracked.size(), trackers_.size());

    for(size_t i = 0; i<trackers_.size();)
    {
        if(!istracked[i])
        {
            istracked.erase(istracked.begin() + i);
            trackers_.erase(trackers_.begin() + i);
        }
        else
            i++;
    }
    tracker_output = trackers_;
    return;
}



void PedestrainTracker::distanceMatrix(std::vector<det_input> feat, cv::Mat &DM,double distThresh)
{
    if (feat.size()==0 || trackers_.size()==0)
        return;

    for (size_t i=0; i<feat.size(); i++)
    {
        for (size_t j=0; j<trackers_.size(); j++)
        {
            cv::Point center = cv::Point(feat[i].x+feat[i].w/2,feat[i].y+feat[i].h/2);
            cv::Point lastCenter = cv::Point((trackers_[j].rect.tl().x+trackers_[j].rect.br().x)/2,(trackers_[j].rect.tl().y+trackers_[j].rect.br().y)/2);
            double distance = sqrt((center.x-lastCenter.x)*(center.x-lastCenter.x)+(center.y-lastCenter.y)*(center.y-lastCenter.y));
            DM.at<float>(i, j)= distance<distThresh?0.1:1;
        }
    }

}


void PedestrainTracker::similarityMatrix(std::vector<std::vector<float>> objFeature,cv::Mat &SM)
{
    if (objFeature.size()==0 || trackers_.size()==0)
    {
        return;
    }

    for (size_t i=0; i<objFeature.size(); i++)
    {
        for (size_t j=0; j<trackers_.size(); j++)
        {
            //float Similarity = cv::norm(objFeature[i],trackers_[j].lastOneFeature,CV_L2);
            float Similarity = cosSimilarity(objFeature[i],trackers_[j].lastOneFeature);
            SM.at<float>(i, j)= Similarity;
        }
    }

}

void PedestrainTracker::genBodyMask(cv::Mat src, std::vector<cv::Rect> feat, std::vector<cv::Mat> &vecMask)
{
    for(int i=0;i<feat.size();i++)
    {
        cv::Mat mask=cv::Mat(src.size(),CV_8UC1,cv::Scalar(0));
        cv::rectangle(mask,feat[i],cv::Scalar::all(255),-1);
//      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1),feat[i].width/4);
        vecMask.push_back(mask);
    }
}

void PedestrainTracker::genHist(cv::Mat src,std::vector<cv::Mat> vecMask,std::vector<cv::Mat> &vecHist)
{
    cv::Mat img = src.clone();


    if(src.channels()==3)
    {
        cv::cvtColor(src,img,CV_BGR2GRAY);
    }

    for(int i=0;i<vecMask.size();i++)
    {

        int histSize = 256;
        float range[] = {0,255};
        const float* histRange={range};

        cv::Mat hist;
        cv::calcHist(&img,1,0,vecMask[i],hist,1,&histSize,&histRange);
        cv::normalize(hist,hist,0,1,cv::NORM_MINMAX,-1,cv::Mat());
        vecHist.push_back(hist);
    }
}

void PedestrainTracker::similarityMatrix(std::vector<cv::Mat> vecHist, cv::Mat &SM)
{
    if (vecHist.size()==0 || trackers_.size()==0)
        return;
    //SM=cv::Mat(feat.size(), info_.size(), CV_32FC1);
    for (size_t i=0; i<vecHist.size(); i++)
    {
        for (size_t j=0; j<trackers_.size(); j++)
        {
            //std::cout << info_[j].lastHist << std::endl;

            double Similarity = cv::compareHist(vecHist[i],trackers_[j].lastHist,CV_COMP_CORREL);

            if((int)Similarity==1)
            {
                Similarity=0;
            }
            SM.at<float>(i, j)= Similarity;
        }
    }

}

void PedestrainTracker::distanceMatrix(std::vector<cv::Rect> feat, cv::Mat &DM)
{
    if (feat.size()==0 || trackers_.size()==0)
        return;

    for (size_t i=0; i<feat.size(); i++)
    {
        for (size_t j=0; j<trackers_.size(); j++)
        {
            cv::Point center = cv::Point((feat[i].tl().x+feat[i].br().x)/2,(feat[i].tl().y+feat[i].br().y)/2);
            cv::Point lastCenter = cv::Point((trackers_[j].rect.tl().x+trackers_[j].rect.br().x)/2,(trackers_[j].rect.tl().y+trackers_[j].rect.br().y)/2);
            double distance = sqrt((center.x-lastCenter.x)*(center.x-lastCenter.x)+(center.y-lastCenter.y)*(center.y-lastCenter.y));
            DM.at<float>(i, j)= distance;
        }
    }
}

