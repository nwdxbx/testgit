#include "tracker.hpp"
#include "graph_min_cut_assign.hpp"
#include "trackNet.h"


object_tracker::object_tracker(){
    objectIDLast = 0;
    fminreportscore = 0.3f;
}



float object_tracker::cosSimilarity(std::vector<float> A, std::vector<float> B)
{
    double sumVecA =0;
    double sumVecB =0;
    double mulAB =0;
    double cosAB =0;

    for(size_t i=0;i<A.size();i++)
    {
        sumVecA+=A[i]*A[i];
        sumVecB+=B[i]*B[i];
        mulAB +=A[i]*B[i];
    }

    sumVecA=sqrt(sumVecA);
    sumVecB=sqrt(sumVecB);

    if((sumVecA-0<0.0001)||(sumVecB-0<0.0001))
    {
        return 0;
    }
    cosAB=mulAB/(sumVecA*sumVecB);
    return cosAB;

}

void object_tracker::getRectScore(std::vector<det_input> objrects, std::vector<double> &rectScore)
{
    //std::vector<double> maxDist;
    for(size_t i=0;i<objrects.size();i++)
    {
        double max=0;
        for(size_t j=0;j<objrects.size();j++)
        {
           cv::Rect rect1 = cv::Rect(objrects[i].x,objrects[i].y,objrects[i].w,objrects[i].h);
           cv::Rect rect2 = cv::Rect(objrects[j].x,objrects[j].y,objrects[j].w,objrects[j].h);
           double dist =similarityTwoFeat(rect1,rect2);

           if(max<dist && dist!=1)
           {
               max=dist;
           }

        }
        rectScore.push_back(1-max);
    }

}

float object_tracker::similarityTwoFeat(cv::Rect &r1, cv::Rect &r2)
{
    int x1 = r1.x;
    int y1 = r1.y;
    int width1 = r1.width;
    int height1 = r1.height;

    int x2 = r2.x;
    int y2 = r2.y;
    int width2 = r2.width;
    int height2 = r2.height;

    int endx = std::max(x1+width1,x2+width2);
    int startx = std::min(x1,x2);
    int width = width1+width2-(endx-startx);

    int endy = std::max(y1+height1,y2+height2);
    int starty = std::min(y1,y2);
    int height = height1+height2-(endy-starty);

    float ratio = 0.0f;
    float Area,Area1,Area2;

    if (width<=0||height<=0)
        return 0.0f;
    else
    {
        Area = width*height;
        Area1 = width1*height1;
        Area2 = width2*height2;
        //ratio = Area /(Area1+Area2-Area);
        ratio = Area /(Area1);
    }

    return ratio;

}

