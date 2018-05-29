#include "zhihui_alg_interface.hpp"
#include "caffe/caffe.hpp"
#include "compare/face_re_id.hpp"
#include "ZBase64/ZBase64.h"
#include "include/version.h"

typedef struct tagZhiHuiHandle
{
    ZhihuiPara para;
    ZBase64 base64;
    int fealens;
}ZhiHuiHandle;

float similarity(const float *attr1, const float *attr2, int lens)
{
    float result =0.0,numerator=0.0,denominator_1=0.0,denominator_2=0.0;
    for(int i = 0; i < lens;i++)
    {
        numerator += attr1[i]*attr2[i];
        denominator_1 += attr1[i]*attr1[i];
        denominator_2 += attr2[i]*attr2[i];
    }
    result = numerator/(sqrt(denominator_1)*sqrt(denominator_2) + FLT_EPSILON);
    return result;
}

/**************************************************************************************************************************************************
函数名：zhihui_getversion
函数功能：获取算法模板版本号
参数说明：
        str:  [out]将当前算法版本号通过str传出
备注：
 **************************************************************************************************************************************************/
void zhihui_getvision(std::string &str)
{
    str = VERSION_MAJOY+VERSION_MINER+VERSION_PATCH;
}

/**************************************************************************************************************************************************
函数名：zhihui_open
函数功能：智慧社区算法初始化函数
参数说明：
        ppvZhihuiHandle: [out]输出智慧社区算法的句柄
        pZhihuiPara: [in]输入智慧社区算法的参数
备注：算法句柄的内存分配在算法内部完成，在close函数中算法内部负责释放
 **************************************************************************************************************************************************/
int zhihui_open(void **ppvZhihuiHandle, ZhihuiPara *pZhihuiPara)
{
    int isize = sizeof(ZhihuiPara);
    if(isize != pZhihuiPara->structsize)
        return EM_ERROR_HEAD_FILE;

    ZhiHuiHandle *pZhihuiHandle = new ZhiHuiHandle();

    int featlens = int(FACE_FEAT_LENS);
    pZhihuiHandle->fealens = featlens;
    pZhihuiHandle->base64.setlen(featlens*sizeof(float));

//    caffe::Caffe::set_mode(caffe::Caffe::GPU);
//    caffe::Caffe::SetDevice(pZhihuiPara->deviceID);

    #undef  GPU
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    #define GPU
    caffe::Caffe::SetDevice(pZhihuiPara->deviceID);

    Face_Re_ID::ins();

    *ppvZhihuiHandle = pZhihuiHandle;
    return EM_SUCESS_STATE;
}

/**************************************************************************************************************************************************
函数名：zhihui_facecal_process
函数功能：输入一张图片，得到人脸的特征
参数说明：
        pvZhihuiHandle: [in]输入智慧社区算法的句柄
        img: [in]输入图片
        res: [out]提取到的人脸特征，如果没有检测到人脸则为空字符串输出，如果有多张人脸则输出最佳一张人脸特征
 **************************************************************************************************************************************************/
int zhihui_facecal_process(void *pvZhihuiHandle, cv::Mat &img, std::string &res)
{
    ZhiHuiHandle *pZhihuiHandle = (ZhiHuiHandle*)pvZhihuiHandle;
    ZBase64 &base64 = pZhihuiHandle->base64;

    std::vector<float> vecres;
    Face_Re_ID::ins().process(img, img, cv::Point(0,0), vecres);

    if(vecres.empty())
        return EM_SUCESS_STATE;
    else if(vecres.size() == pZhihuiHandle->fealens)
        res = base64.Encode(reinterpret_cast<const unsigned char *>(vecres.data()),vecres.size() * sizeof(float));
    else
        return EM_ERROR_FEAT_LENS;

    return EM_SUCESS_STATE;
}

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
int zhihui_facecompare_process(std::string &feature1, std::string &feature2, float &fsimilarity)
{
    //ZhiHuiHandle *pZhihuiHandle = (ZhiHuiHandle*)pvZhihuiHandle;
    ZBase64 base64;
    //ZBase64 &base64 = pZhihuiHandle->base64;

    int OutByte = 0;

//    if(feature1.length() != pZhihuiHandle->base64.fealength)
//        return EM_ERROR_FEAT_LENS;
//    if(feature2.length() != pZhihuiHandle->base64.fealength)
//        return EM_ERROR_FEAT_LENS;


    std::vector<float> res1, res2;
    std::string decode1 = base64.Decode(feature1.c_str(), feature1.length(), OutByte);
    std::string decode2 = base64.Decode(feature2.c_str(), feature2.length(), OutByte);

    const float *f1 = reinterpret_cast<const float*>(decode1.data());
    const float *f2 = reinterpret_cast<const float*>(decode2.data());

    //fsimilarity = similarity(f1, f2, pZhihuiHandle->fealens);
    fsimilarity = similarity(f1, f2, int(FACE_FEAT_LENS));

    return EM_SUCESS_STATE;
}

/**************************************************************************************************************************************************
函数名：zhihui_close
函数功能：释放资源
参数说明：
        pvzhihuiHandle：[in]通过句柄将open分配的资源释放
备注：
 **************************************************************************************************************************************************/
int zhihui_close(void *pvZhihuiHandle)
{
    return EM_SUCESS_STATE;
}
