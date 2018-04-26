/********************************************************************************* 
  *Copyright(C)
  *FileName: plateApi.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2017-12-26
  *Description:  interface of plateApi
  *History:
   Date: 2017-12-26
   Author: Jin
   Modification: No
**********************************************************************************/
#include "plateApi.hpp"

//test
#include<boost/format.hpp>

#define DETECT_LEN (7)

//test
static int g_num = 1;

bool ScoreSort(vector<float> v1, vector<float> v2)
{
    return v1[2] > v2[2];
}


int DetectAndRecognizePlate(Detector& detector, Classifier& classifier, const cv::Mat& img, RESULT_PLATE& stResult)
{
	std::string plateNum;
    long long time_begin;
    long long time_end;
    int iRet = 0;

    time_begin = cvGetTickCount();
    std::vector<vector<float> > detections = detector.Detect(img);
    time_end = cvGetTickCount();
    //printf("detect time cost = %f\n", (time_end - time_begin) / cvGetTickFrequency() / 1000000);

    std::sort(detections.begin(), detections.end(), ScoreSort);
    /* Print the detection results. */
    for (int i = 0; i < detections.size() && i < 3; ++i)
    {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), DETECT_LEN);
        const float score = d[2];
        int label = int(d[1]);

        //std::cout<<"label is :"<<label<<std::endl;
        if ((ELogoType)label == eCHEPAI)
        {
            cv::Mat imgOrg = img.clone();
            //std::cout << "plate_score=" << score << std::endl;
            if ((d[3] < 0) || (d[4] < 0) || (d[5] < 0) || (d[6] < 0) || (d[3] > d[5]) || (d[4] > d[6]))
            {
                std::cout << "[ERROR]: " << d[3] << " " << d[4] << " " << d[5] << " " << d[6] << std::endl;
                return -1;
            }
            cv::Rect rect((d[3] * img.cols),(d[4] * img.rows),(d[5] * img.cols -d[3] * img.cols),(d[6] * img.rows-d[4] * img.rows));
            if(rect.width+rect.x >= imgOrg.cols){
                rect.width = imgOrg.cols-rect.x-1;
            }
            if(rect.height+rect.y >= imgOrg.rows)
            {
                rect.height = imgOrg.rows-rect.y-1;
            }
            cv::Mat imgPlate=imgOrg(rect).clone();
            //test
            //cv::rectangle(imgOrg, rect, cv::Scalar(0,0,255));
            //cv::imshow("plate_detect", imgOrg);
            //cv::waitKey(0);
            //

            iRet = classifier.RecognizePlate(imgPlate, stResult);
            if (iRet == -1)
            {
                std::cout << "[ERROR]: RecognizePlate faild!!" << std::endl;
                return -1;
            }
            stResult.rect = rect;
            //test
//            string str_num;
//            str_num = boost::str(boost::format("%05d")%g_num);
//            std::string strPLate = "/work/project/tuzhen/bin/result5/"+str_num+"_"+stResult.plateNum+"_0"".jpg";
//            cout<<"strPLate="<<strPLate<<endl;
//            cv::imwrite(strPLate, imgPlate);
//            g_num++;
//            std::cout<<"num="<<g_num<<std::endl;
            //
            break;
        }
    }

    return 1;
}
