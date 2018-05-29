/**********************************************************************************************************************************************
module name:  zhihui_alg_interface.hpp
function: face recognition, plate ocr recognition
remark: not support multithread now, resources created by one thread must process and release in the same thread.

Vesion：V1.0  Copyright(C) 2018-2022 em-data, All rights reserved.
-------------------------------------------------------------------------------
modify record:
date                 version     author                   state
2018/05/02           1.0	     zhfyuan                  create
************************************************************************************************************************************************/

#ifndef __ZHIHUI_ALG_INTERFACE_HPP__
#define __ZHIHUI_ALG_INTERFACE_HPP__

//智慧社区算法参数
typedef struct tagZhihuiPara
{
    int structsize;//自身结构体大小
    int deviceID;//gpu ID号   
}ZhihuiPara;

//函数异常执行返回错误码
typedef enum
{
    EM_SUCESS_STATE=0,  //成功执行
    EM_ERROR_MALLOC,   //分配内存失败
    EM_ERROR_INPUT_PARA, //输入参数不合理
    EM_ERROR_HEAD_FILE //头文件版本不匹配
}EReturn_Errcode;


#include <opencv2/opencv.hpp>
/**************************************************************************************************************************************************
函数名：zhihui_open
函数功能：智慧社区算法初始化函数
参数说明：
        ppvZhihuiHandle: [out]输出智慧社区算法的句柄
        pZhihuiPara: [in]输入智慧社区算法的参数
备注：算法句柄的内存分配在算法内部完成，在close函数中算法内部负责释放
 **************************************************************************************************************************************************/ 
int zhihui_open(void **ppvZhihuiHandle, ZhihuiPara *pZhihuiPara);

/**************************************************************************************************************************************************
函数名：zhihui_facecal_process
函数功能：输入一张图片，得到人脸的特征
参数说明：
        pvZhihuiHandle: [in]输入智慧社区算法的句柄
        img: [in]输入图片        
        res: [out]提取到的人脸特征，如果没有检测到人脸则为空字符串输出，如果有多张人脸则输出最佳一张人脸特征
 **************************************************************************************************************************************************/
int zhihui_facecal_process(void *pvZhihuiHandle, cv::Mat &img, std::string &res);

/**************************************************************************************************************************************************
函数名：zhihui_facecompare_process
函数功能：输入两张人脸特征，得到两者的相似度，范围[0,1]
参数说明：
        pvZhihuiHandle: [in]输入智慧社区算法的句柄
        feature1: [in]人脸1的特征        
        feature2: [in]人脸2的特征
        fsimilarity: [out]人脸1和人脸2的相似度
 备注：两个特征维度必须相等
 **************************************************************************************************************************************************/
int zhihui_facecompare_process(void *pvZhihuiHandle, std::string &feature1, std::string &feature2, float &fsimilarity);

/**************************************************************************************************************************************************
函数名：zhihui_close
函数功能：释放资源
参数说明：
        pvzhihuiHandle：[in]通过句柄将open分配的资源释放
备注：
 **************************************************************************************************************************************************/
int zhihui_close(void *pvZhihuiHandle);

#endif