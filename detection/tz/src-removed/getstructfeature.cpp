#include "getstructfeature.hpp"


void attributeAnalysis(cv::Mat &img,std::vector<cv::Vec3f> &pts,cv::Rect &box,PersonInfo &perInfo)
{
    det_res res;
    humanAttribute_caffe::ins().process(img,res);
    perInfo.sleeve_length = static_cast<EPerson_Clothes_Lenght>(res.sleeveLength);
    perInfo.pants_length = static_cast<EPerson_Clothes_Lenght>(res.pantsLength);
    perInfo.sleeve_length_score = res.sleeveLengthScore;
    perInfo.pants_length_score = res.pantsLengthScore;
    perInfo.coat_color = static_cast<EPerson_Clothes_Color>(res.upclsColor);
    perInfo.trouser_color = static_cast<EPerson_Clothes_Color>(res.downclsColor);
    perInfo.coat_color_score = res.coat_color_score;
    perInfo.trouser_color_score = res.trouser_color_score;

}

//int getstructfeatureopen(TGetFeature **ppstGetFeature, int flag)
//{
//    TGetFeature *ptGetFeature = (TGetFeature*)malloc(sizeof(TGetFeature));
//    ptGetFeature->flag = flag;

//    *ppstGetFeature = ptGetFeature;

//    //ins();

//    return 0;
//}

//int getstructfeatureprocess(TGetFeature *pstGetFeature, PersonInfo &stPersonInfo)
//{
//    //ins().process
//    return 0;
//}

//void getstructfeatureclose(TGetFeature *pstGetFeature)
//{
//    if(pstGetFeature)
//    {
//        free(pstGetFeature);
//        pstGetFeature = NULL;
//    }
//    return;
//}
