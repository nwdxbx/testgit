#include "track.hpp"
#include "graph_min_cut_assign.hpp"

object_tracker::object_tracker(){
    objectIDLast = 0;
    fminreportscore = 0.3f;
}


void object_tracker::process(cv::Mat srcImg, std::vector< vector<cv::Vec3f> > &points, std::vector<tracker_obj_info> &tracker_output)
{
    const int interval = 10;
    const int static_interval = 30;

    //step0: transfer the points to rects
    std::vector<cv::Rect> objrects;
    std::vector<vector<cv::Vec3f> > newpoints;

    if(!points.empty())
    {
        for(size_t i=0; i<points.size(); i++)
        {
            cv::Point lt(10000,10000), rb(-10000,-10000);
            for (size_t j=0; j<points[i].size(); j++)
            {
                if (points[i][j][2]>0.2f)
                {
                    lt=cv::Point(std::min(lt.x, (int)points[i][j][0]), std::min(lt.y, (int)points[i][j][1]));
                    rb=cv::Point(std::max(rb.x, (int)points[i][j][0]), std::max(rb.y, (int)points[i][j][1]));
                }
            }
            //only accept the object rect bigger than 10x15
            if(rb.x - lt.x > 10 && rb.y - lt.y > 15)
            {
                objrects.push_back(cv::Rect(lt, rb));
                newpoints.push_back(points[i]);
            }
        }
    }

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

            info_one.fcurscore = getScore(points[i], objrects[i]);

            info_one.lastHist = vecHist[i];
            info_one.objectID = objectIDLast + 1;
            info_one.trackerlife = 1;        
            info_one.rect = objrects[i];
            info_one.points = newpoints[i];
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
            tracker_match.points = newpoints[i];
            tracker_match.lastHist = vecHist[i];
            tracker_match.fcurscore = getScore(points[i],objrects[i]);
            if(tracker_match.fcurscore > tracker_match.fmaxscore && tracker_match.fcurscore > object_tracker::fminreportscore)
                tracker_match.breport = true;
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


void object_tracker::genBodyMask(cv::Mat src, std::vector<cv::Rect> feat, std::vector<cv::Mat> &vecMask)
{
    for(int i=0;i<feat.size();i++)
    {
        cv::Mat mask=cv::Mat(src.size(),CV_8UC1,cv::Scalar(0));
        cv::rectangle(mask,feat[i],cv::Scalar::all(255),-1);
//      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1),feat[i].width/4);
        vecMask.push_back(mask);
    }
}

void object_tracker::genHist(cv::Mat src,std::vector<cv::Mat> vecMask,std::vector<cv::Mat> &vecHist)
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

void object_tracker::similarityMatrix(std::vector<cv::Mat> vecHist, cv::Mat &SM)
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

void object_tracker::distanceMatrix(std::vector<cv::Rect> feat, cv::Mat &DM)
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

double object_tracker::getScore(std::vector<cv::Vec3f> opts,cv::Rect rect)
{
   double probSum=0.0;
   int count=0;
   for(int i=0;i<opts.size();i++)
   {
       if(opts[i][2]>0)
       {
           probSum+=opts[i][2];
           count++;
       }

   }
   double probMean=probSum/count;
   double countScore=count/18.0;

   double ratio = (double)rect.height/(double)rect.width;
   double ratioScore=1;
   if(ratio<2.0)
   {
       ratioScore=ratio/2.0;
   }
   else if(ratio>2.0)
   {
       ratioScore=2.0/ratio;
   }


   return probMean*countScore*ratioScore;

}



//void object_tracker::process(std::vector< cv::Rect > &feat, std::vector<bool> &shouldAdd)
//{
////    const int interval = 50;
////    const int static_interval = 150;

//    const int interval = config::ins().get_interval();
//    const int static_interval = config::ins().get_static_interval();

//    //step1: compute SM Matrix, M X N, M: observe dim; N: tracker dim
//    cv::Mat SM=cv::Mat(feat.size(), info_.size(), CV_32F);

//    similarityMatrix(feat,SM);

//    //step2: graphMinCut to get if the observe rect matched or not, assign dim: M
//    //if matched, assign value is the tracker index, else -1
//    std::vector<int> assign;
//    if(!SM.empty())
//    {
//        GraphAssign::graphMinCut(SM, assign);
//        std::cout << SM << std::endl;
//        for(size_t i=0; i<SM.rows; i++)
//        {
//            //if(assign[i] != -1 && SM.at<float>(i,assign[i]) > 1-0.25f)
//            if(assign[i] != -1 && SM.at<float>(i,assign[i]) > 1-0.05f)
//            {
//                assign[i] = -1;
//            }
//        }
//    }
//    else
//    {
//        for(auto &i : feat)
//        {
//            assign.push_back(-1);
//        }
//    }

//    //step3: add new tracker
//    std::vector<bool> istracked;
//    for(auto &i : info_)
//    {
//        istracked.push_back(false);
//    }

//    for(size_t i=0; i<assign.size(); i++)
//    {
//        //not match, add new tracker
//        if(assign[i] == -1)
//        {
//            tracker_obj_info info_one;
//            info_one.rect = feat[i];
//            info_one.lastRect = feat[i];
//            info_one.count = 1;
//            info_one.objectID = objectIDLast + 1;
//            objectIDLast++;
//            shouldAdd.push_back(true);
//            info_.push_back(info_one);
//            istracked.push_back(true);
//        }
//        else
//        {
//            tracker_obj_info &info_matchone = info_[assign[i]];
//            updateTracker(feat[i], info_matchone);
//            info_matchone.count++;
//            istracked[assign[i]] = true;
//            //static object will not report
////            if(info_matchone.count % interval == 0 && similarityTwoFeat(info_matchone.rect, info_matchone.lastRect) < 0.3f)
////            {
////                shouldAdd.push_back(true);
////                info_matchone.lastRect = info_matchone.rect;
////            }
////            else
////            {
////                shouldAdd.push_back(false);
////            }

//            bool breport = false;
//            if(info_matchone.count % interval == 0 && similarityTwoFeat(info_matchone.rect, info_matchone.lastRect) < 0.3f)
//            {
//                breport = true;
//                info_matchone.lastRect = info_matchone.rect;
//            }
//            if(info_matchone.count % static_interval == 0)
//                breport = true;

//            shouldAdd.push_back(breport);

//        }
//    }

//    //step4: delete tracker that not match
//    CHECK_EQ(istracked.size(), info_.size());

//    for(size_t i = 0; i<info_.size();)
//    {
//        if(!istracked[i])
//        {
//            istracked.erase(istracked.begin() + i);
//            info_.erase(info_.begin() + i);
//        }
//        else
//            i++;
//    }
//}

//float object_tracker::similarityTwoFeat(cv::Rect &r1, cv::Rect &r2)
//{

//    int x1 = r1.x;
//    int y1 = r1.y;
//    int width1 = r1.width;
//    int height1 = r1.height;

//    int x2 = r2.x;
//    int y2 = r2.y;
//    int width2 = r2.width;
//    int height2 = r2.height;

//    int endx = std::max(x1+width1,x2+width2);
//    int startx = std::min(x1,x2);
//    int width = width1+width2-(endx-startx);

//    int endy = std::max(y1+height1,y2+height2);
//    int starty = std::min(y1,y2);
//    int height = height1+height2-(endy-starty);

//    float ratio = 0.0f;
//    float Area,Area1,Area2;

//    if (width<=0||height<=0)
//        return 0.0f;
//    else
//    {
//        Area = width*height;
//        Area1 = width1*height1;
//        Area2 = width2*height2;
//        ratio = Area /(Area1+Area2-Area);
//    }

//    return ratio;
//}

//void object_tracker::similarityMatrix(std::vector< cv::Rect > &feat, cv::Mat &SM)
//{
//    if (feat.size()==0 || info_.size()==0)
//        return;
//    //SM=cv::Mat(feat.size(), info_.size(), CV_32FC1);
//    for (size_t i=0; i<feat.size(); i++)
//    {
//        for (size_t j=0; j<info_.size(); j++)
//        {
//            float dist=similarityTwoFeat(feat[i], info_[j].rect);
//            SM.at<float>(i, j)=1-dist;
//        }
//    }
//}

//void object_tracker::updateTracker(const cv::Rect &obsRect, tracker_obj_info &tracker)
//{
//    float learningRate = 0.1;
//    //hard code
//    tracker.rect = obsRect;
//}
