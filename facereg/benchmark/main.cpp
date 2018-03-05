#include <iostream>
#include <glog/logging.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <limits>
#include "face_reid_caffe.hpp"

void clipBox(cv::Rect &rect, cv::Size imgsz)
{
    cv::Rect k;
    k.x = std::min(std::max(0, rect.x), imgsz.width-1);
    k.y = std::min(std::max(0, rect.y), imgsz.height-1);
    k.width = std::min(std::max(0, rect.x+rect.width), imgsz.width)-k.x;
    k.height = std::min(std::max(0, rect.y+rect.height), imgsz.height)-k.y;

    rect=k;
}

void get_face_feat(cv::Mat &img, std::vector<float> &face)
{
    face.clear();
    if (img.empty())
        return;
        
    cv::Mat faceIm=img.clone();
    alg::face_reid_caffe::ins().process(faceIm, face);
    
    cv::Mat mfaceIm;
    cv::flip(faceIm, mfaceIm, 1);
    std::vector<float> mface;
    alg::face_reid_caffe::ins().process(mfaceIm, mface);
    face.insert(face.end(), mface.begin(), mface.end());
    //for (size_t i=0; i<mface.size(); i++)
    //{
    //    face[i]+=mface[i];
    //}

}

float get_dist(std::vector<float> &feat1, std::vector<float> &feat2)
{
    if (feat1.size() != feat2.size())
        return -1;
    if (feat1.size() == 0)
        return -1;   
    float sum1=0.f, sum2=0.f, sum3=0.f;
    for (size_t i=0; i<feat1.size(); i++)
    {
        //LOG(INFO) << feat1[i] << " ";
        sum1+=feat1[i]*feat1[i];
        sum2+=feat2[i]*feat2[i];
        sum3+=feat1[i]*feat2[i];
    }
    //LOG(INFO) << std::endl;
    return sum3/std::max(std::sqrt(sum1*sum2), 1e-6f);
}

float get_dist2(std::vector<float> &feat1, std::vector<float> &feat2)
{
    if (feat1.size() != feat2.size())
        return -1;
    if (feat1.size() == 0)
        return -1;   
    float sum1=0.f, sum2=0.f, sum3=0.f;
    for (size_t i=0; i<feat1.size(); i++)
    {
        //LOG(INFO) << feat1[i] << " ";
        sum1+=feat1[i]*feat1[i];
        sum2+=feat2[i]*feat2[i];
    }
    //LOG(INFO) << "len " << sum1;
    for (size_t i=0; i<feat1.size(); i++)
    {
        //LOG(INFO) << feat1[i] << " ";
        sum3+=(feat1[i]/std::max(sqrt(sum1), 1e-6)-feat2[i]/std::max(sqrt(sum2), 1e-6))*(feat1[i]/std::max(sqrt(sum1), 1e-6)-feat2[i]/std::max(sqrt(sum2), 1e-6));
    }
    //LOG(INFO) << std::endl;
    return sum3;
}

float get_res(const std::vector< std::pair<float, int> > &dist_all)
{
    std::vector<float> thresh;
    for (float i=0.f; i<4.f; i+=0.001f)
    {
        thresh.push_back(i);
    }

    float max_acc = std::numeric_limits<float>::lowest();
    float max_tpr, max_fpr;
    float maxthresh = 0.f;
    for (size_t i=0; i<thresh.size(); i++)
    {
        int tp=0, fp=0, tn=0, fn=0;        
        for (size_t j=0; j<dist_all.size(); j++)
        {
            tp+=dist_all[j].first<=thresh[i] && dist_all[j].second==1;
            fp+=dist_all[j].first<=thresh[i] && dist_all[j].second==0;
            tn+=dist_all[j].first>thresh[i] && dist_all[j].second==0;
            fn+=dist_all[j].first>thresh[i] && dist_all[j].second==1;
        }
        float tpr = (tp+fn)==0 ? 0 : float(tp)/float(tp+fn);
        float fpr = (fp+tn)==0 ? 0 : float(fp)/float(fp+tn);
        float acc = float(tp+tn)/dist_all.size();
        if (acc > max_acc)
        {
            max_acc = acc;
            maxthresh = thresh[i];
        }
    }
    LOG(INFO) << "Thresh is: " << maxthresh;
    return max_acc;
}

int main(int argc, char *argv[])
{
//    std::ifstream fid(argv[1]);
    
    std::ifstream fid("/media/d/FaceRecognition/benchmark/benchmark.txt");
    std::string line;
    std::vector<std::pair<float, int> > dist_all;
    while (std::getline(fid, line))
    {
        int sp1 = line.find(' ');
        int sp2 = line.rfind(' ');
        std::string imgname1=line.substr(0, sp1);
        std::string imgname2=line.substr(sp1+1, sp2-sp1-1);
        std::string issame=line.substr(sp2+1);
        int same;
        std::stringstream ss(issame);
        ss >> same;
        LOG(INFO) << imgname1 << "---" << imgname2 << "---" << issame;
        cv::Mat img1=cv::imread(imgname1);
        cv::Mat img2=cv::imread(imgname2);
        
        double start, end;
        std::vector<float> face1, face2;
        start = cv::getTickCount();
        get_face_feat(img1, face1);
        end = cv::getTickCount();
        //LOG(INFO) << "time using : " << (end-start) / cv::getTickFrequency();
        get_face_feat(img2, face2);
        
        float score=get_dist2(face1, face2);
        LOG(INFO) << "Score: " << score;
        dist_all.push_back(std::make_pair(score, same));
    }
    
    float acc = get_res(dist_all);
    LOG(INFO) << "The accuracy is: " << acc;

    return 0;
}
















