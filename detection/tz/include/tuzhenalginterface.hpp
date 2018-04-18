/************************************************************************************
module name:  tuzhenalgInterface.h
function: imput video stream, output objects with property

Vesion：V1.0  Copyright(C) 2016-2020 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author              state
2017/06/01           1.0	     mx                  create
2017/09/12           2.0	     zhfyuan		     c++ -> c
**************************************************************************************/
#ifndef TUZHENALGINTERFACE_HPP
#define TUZHENALGINTERFACE_HPP
#include "common.h"


//static int head_count=0;

//batchsize = 1
//视频处理中每一帧的输入信息
typedef struct tagTuzhenInput
{
    int cameraID; //由于每个实例会处理多路视频，这里以camereID号做区分
    Mat img;      //视频流中一帧图像
    ESource_Type sourceType= EM_SOURCE_IMAGE;     //
    float fintegritythresh=0.0f; //人形完整度阈值
    bool useROI = false;
    cv::Rect roi;
}TuzhenInput;

//batchsize > 1, 可以加速
//typedef struct tagTuzhenInput
//{
//    vector<int> cameraIDs;
//    vector<Mat> imgs;
//}TuzhenInput;

//视频处理中每一帧的输出信息
typedef struct tagTuzhenOutput
{
    vector<PersonInfo> personvec; //多个行人的位置及各种属性
    EPersons_Behavior_Type ePersonstype; //人群是否有异常行为
    vector<VehicleInfo>vehicles_info;
    CrowdEstimateInfo crowd_estimate_info;
}TuzhenOutput;


//单张图片输入，结合标志位信息。目前只支持目标占满图片且只有一个人情况
typedef struct tagTzImageInput
{
    int flag; //0:行人特征; 1：人脸特征; 2：行人特征+行人属性; 3：人脸特征+人脸属性
    Mat img;      //一帧图像
}TzImageInput;

typedef struct tagTuzhenPara
{
    int structsize;
    int deviceID; //仅仅在tuzhen_open函数中有效
    int intervalFrames; //算法每隔多少帧调用一次，用来确定算法内部的一些阈值
    int itvreportmove; //运动目标每调用算法多少次上报
    int itvreportstatic; //静止目标每调用算法多少次上报
    float face_clarity_thresh = 6500;
    int face_min_width = 55;
    int face_min_height = 65;
    float mask_rcnn_prob_thresh = 0.8;
    int eoptfun; //确定哪些算法功能需要开启
}TuzhenPara;

/**************************************************************************************************************************************************
函数名：get_version
函数功能：获取算法模板版本号
参数说明：
        str:  [out]将当前算法版本号通过str传出
备注：
 **************************************************************************************************************************************************/
void get_version(std::string &str);


/**************************************************************************************************************************************************
函数名：tuzhen_open
函数功能：图侦算法初始化函数
参数说明：
        ppvTuzhenHandle: [out]输出图侦算法的句柄
        pTuzhenPara: [in]输入图侦算法的参数
备注：算法句柄的内存分配在算法内部完成，在close函数中算法内部负责释放
 **************************************************************************************************************************************************/
int tuzhen_open(void **ppvTuzhenHandle, TuzhenPara *pTuzhenPara);


/**************************************************************************************************************************************************
函数名：tuzhen_process
函数功能：图侦算法的主处理函数
参数说明：
        pvTuzhenHandle: [in]输入图侦算法的句柄
        tzInput: [in]输入单帧图片和摄像机ID号
        tzOutput: [out]所有行人属性信息以及当前视频状态信息
备注：为了缓解数据库压力，行人属性中包含一个是否上报变量，只有上报的目标才添加进数据库。
 **************************************************************************************************************************************************/
int tuzhen_process(void *pvTuzhenHandle, TuzhenInput &tzInput, TuzhenOutput &tzOutput);

/**************************************************************************************************************************************************
函数名：tuzhen_setpara
函数功能：图侦算法
参数说明：
        pvTuzhenHandle: [in]输入图侦算法的句柄
        pTuzhenPara: [in]图侦算法的参数，可以控制功能项的开启
备注：在算法运行过程中可以随时调用该函数，不需要重启算法模块。唯有一个例外就是gpu的设备ID只能在open函数中设置。
 **************************************************************************************************************************************************/
int tuzhen_setpara(void *pvTuzhenHandle, TuzhenPara *pTuzhenPara);


/**************************************************************************************************************************************************
函数名：feat_cal
函数功能：图侦算法的主处理函数
参数说明：
        pvTuzhenHandle: [in]输入图侦算法的句柄
        tzInput: [in]输入单帧图片和摄像机ID号
        tzOutput: [out]所有行人属性信息以及当前视频状态信息
备注：为了缓解数据库压力，行人属性中包含一个是否上报变量，只有上报的目标才添加进数据库。
 **************************************************************************************************************************************************/
//int tuzhen_cal_feat(void *pvTuzhenHandle, Mat &img, TuzhenOutput &tzOutput);
int feat_cal(cv::Mat &img, EFeature_Cal_Type flag, std::string&res);//模型加载放到tuzhen_open同一个线程里边去进行;

//int feat_cal_roi(cv::Mat&img,cv::Rect rt,EFeature_Cal_Type flag, std::string&res);

int feature_cal_roi(cv::Mat &img, cv::Rect rt,EFeature_Cal_Type flag, std::string&res);//模型加载放到tuzhen_open同一个线程里边去进行;

int face_feature_attri(cv::Mat &img,PersonInfo& personinfo);//for face only;

int remove_camera_by_id(void *pvTuzhenHandle,int cameraId);//删除特定标号的摄像头

/**************************************************************************************************************************************************
函数名：tuzhen_close
函数功能：释放资源
参数说明：
        pvTuzhenHandle：[in]通过句柄将open分配的资源释放
备注：
 **************************************************************************************************************************************************/
void tuzhen_close(void *pvTuzhenHandle);

class PedestrainTracker;
class VehicleTracker;
typedef struct tagTuzhenalgHandle
{
    int  deviceID;
    int  eoptfun = EM_OPTIONAL_ITEM_PEDESTRAIN;
    std::map<int, PedestrainTracker*> pedestrain_tracker;
    std::map<int, VehicleTracker*> vehicle_tracker;
    float face_clarity_thresh = 6500;
    int face_min_width = 55;
    int face_min_height = 65;
    float mask_rcnn_prob_thresh = 0.8;
    //base featuremap
    //caffe::Blob *pBaseBlob;
    //compute feature
    //TGetFeature *pGetFeature;
}TuzhenalgHandle;




#endif // TUZHENALGINTERFACE_HPP
