#include "tuzhenfeature.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "ZBase64.h"

float compute_similarity(cv::Mat image1, cv::Mat image2);
int rsv_get_video_feature(void *pvRsvHandle, std::string videopath, std::string& feature);

typedef struct tagTRemoveHandle
{
    int img_width;
    int img_height;
    int crop_height;
    float f32similar_thresh;
    ZBase64 base64;
    std::map<std::string, std::string> map_uuid2fea;
}TRemoveHandle;



int rsv_open(void **ppvRsvHandle, float fsimthresh, EVideofeature_Load_Mode flag, int feanum, int deviceID)
{

    TRemoveHandle *pRsvHandle;
    pRsvHandle = new TRemoveHandle();

    pRsvHandle->img_width = 64;
    pRsvHandle->img_height = 64;

    //pRsvHandle->crop_height = 16;
    pRsvHandle->crop_height = 0;
    pRsvHandle->f32similar_thresh = fsimthresh;

    pRsvHandle->base64.setlen(pRsvHandle->img_width * (pRsvHandle->img_height - pRsvHandle->crop_height));

    if(pRsvHandle == NULL)
        return EM_ERROR_MALLOC;

    *ppvRsvHandle = pRsvHandle;
    //only cpu availble
    return EM_SUCESS_STATE;
}

int rsv_setpara(void *pvRsvHandle, float fsimthresh)
{
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;
    pRsvHandle->f32similar_thresh = fsimthresh;
    return EM_SUCESS_STATE;
}

int rsv_loadfeature(void *pvRsvHandle, std::string userdata, std::string feature)
{
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;
    std::map<std::string, std::string>& map_uuid2fea = pRsvHandle->map_uuid2fea;
    ZBase64& base64 = pRsvHandle->base64;
    int img_width = pRsvHandle->img_width;
    int real_height = pRsvHandle->img_height - pRsvHandle->crop_height;

    int lens = feature.size();

    if(lens == base64.fealength*2)
    {
        int OutByte = 0;
        bool bRet = map_uuid2fea.count(userdata);
        if(bRet)
            return EM_ERROR_INPUT;

        map_uuid2fea[userdata] = feature;
        return EM_SUCESS_STATE;
    }
    else
    {
        return EM_ERROR_INPUT;
    }
}

int rsv_checkvideo(void *pvRsvHandle, std::string inputvideopath, std::string& feature, std::string& sameuserdata)
{
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;
    std::map<std::string, std::string>& map_uuid2fea = pRsvHandle->map_uuid2fea;
    ZBase64& base64 = pRsvHandle->base64;
    sameuserdata.clear();

    std::string sub1, sub2;
    if(feature.empty())
    {
        rsv_get_video_feature(pvRsvHandle, inputvideopath, feature);
        if(feature.empty())
            return EM_ERROR_INPUT;
    }

    int lens = feature.size();
    sub1 = feature.substr(0, lens/2);
    sub2 = feature.substr(lens/2, lens/2);

    int OutByte = 0;
    int img_width = pRsvHandle->img_width;
    int real_height = pRsvHandle->img_height - pRsvHandle->crop_height;

    string back1 = base64.Decode(sub1.c_str(), sub1.length(), OutByte);
    cv::Mat image1(img_width, real_height, CV_8U);
    image1.data = (uchar*)back1.c_str();

    string back2 = base64.Decode(sub2.c_str(), sub2.length(), OutByte);
    cv::Mat image2(img_width, real_height, CV_8U);
    image2.data = (uchar*)back2.c_str();

    float f32similarity;

    for(auto iter = map_uuid2fea.begin(); iter != map_uuid2fea.end(); iter++)
    {
        int OutByte = 0;
        //cout << iter->first << " : " << iter->second << endl;
        sub1 = iter->second.substr(0, base64.fealength);
        sub2 = iter->second.substr(base64.fealength, base64.fealength);
        string libback1 = base64.Decode(sub1.c_str(), sub1.length(), OutByte);
        string libback2 = base64.Decode(sub2.c_str(), sub2.length(), OutByte);

        cv::Mat imagelib(img_width, real_height, CV_8U);
        imagelib.data = (uchar*)libback1.c_str();

        cv::Mat imagelib2(img_width, real_height, CV_8U);
        imagelib2.data = (uchar*)libback2.c_str();

       f32similarity = compute_similarity(image1, imagelib);
       if(f32similarity > pRsvHandle->f32similar_thresh)
       {
           sameuserdata = iter->first;
           return EM_SUCESS_STATE;
       }

       f32similarity = compute_similarity(image2, imagelib2);
       if(f32similarity > pRsvHandle->f32similar_thresh)
       {
           sameuserdata = iter->first;
           return EM_SUCESS_STATE;
       }
    }

    return EM_SUCESS_STATE;
}

int rsv_delvideo(void *pvRsvHandle, std::string userdata)
{
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;
    std::map<std::string, std::string>& map_uuid2fea = pRsvHandle->map_uuid2fea;

    auto key = map_uuid2fea.find(userdata);
    if(key!=map_uuid2fea.end())
    {
       map_uuid2fea.erase(key);
       return EM_SUCESS_STATE;
    }
    return EM_ERROR_INPUT;
}

void rsv_close(void *pvRsvHandle)
{
    delete (TRemoveHandle*)pvRsvHandle;
}


float compute_similarity(cv::Mat image1, cv::Mat image2)
{
    const int channels[1] = {0};
    const int histSize  = 32;
    float hrange[2] = {0,255};
    const float *range[1] = {hrange};
    const float weight[32] = {1.0, 1.5, 2.1, 3.0, 4.4, 6.4, 9.3, 13.5,
                              19.5, 28.2, 41, 59, 86, 124, 181, 262,
                              380, 550, 800, 1160, 1680, 2440, 3540, 5120,
                              7430, 10000, 15000, 23000, 33000, 47500, 69000, 100000};
    float totalnum = 0;
    float firstnum = 0;

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    //std::cout << diff << std::endl;

    //double minv = 0.0, maxv = 0.0;
    //double* minp = &minv;
    //double* maxp = &maxv;
    //cv::minMaxIdx(diff,minp,maxp);

    cv::Mat hist;
    cv::calcHist(&diff, 1, channels, Mat(), hist, 1, &histSize, range);

    float* p = hist.ptr<float>(0);
    firstnum = p[0];
    for(size_t i=0; i<histSize; ++i)
    {
        totalnum += p[i]*weight[i];
    }

    return firstnum/(totalnum+0.0001f);
}

int rsv_get_video_feature(void *pvRsvHandle, std::string videopath, std::string& feature)
{
    feature.clear();
    cv::VideoCapture cap;
    cv::Mat frame_begin, frame_end;
    cv::Mat mv[3];
    int frame_number;

    if(!cap.open(videopath))
        return EM_ERROR_INPUT;
    cap >> frame_begin;
    if(frame_begin.empty() || frame_begin.rows <=0 || frame_begin.cols <=0)
        return EM_ERROR_INPUT;

    cv::Mat frame_begin_ss, frame_end_ss;
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;

    int img_width = pRsvHandle->img_width;
    int img_height = pRsvHandle->img_height;
    ZBase64 &base64 = pRsvHandle->base64;

    if(frame_begin.channels() == 3)
    {
        //INTER_AREA和INTER_LINEAR两种方式会造成后面计算得分差别很大，因为如果选择INTER_LINEAR，不是完全相同的视频（除了视频格式）差别会更大
        cv::split(frame_begin, mv);
        cv::resize(mv[0], frame_begin_ss, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        cv::resize(frame_begin, frame_begin_ss, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
    }

    frame_number = 0;

    frame_number = cap.get(cv::CAP_PROP_FRAME_COUNT);

    if(frame_number > 3)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_number-1);//从此时的帧数开始获取帧
        cap >> frame_end;

        if(frame_end.empty() || frame_end.rows <=0 || frame_end.cols <=0)
            return EM_ERROR_INPUT;

        if(frame_end.channels() == 3)
        {

            cv::split(frame_end, mv);
            cv::resize(mv[0], frame_end_ss, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
        }
        else
        {
            cv::resize(frame_end, frame_end_ss, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);
        }
        //std::vector<uchar> data_encode;
        //cv::imencode(".png", frame_begin_ss, data_encode);
        //std::cout << data_encode.size() << std::endl;

        //zhfyuan 2018 64 -> width height data+offset
        std::string imgbase64_begin = base64.Encode(frame_begin_ss.data, img_width*img_height);
        std::string imgbase64_end = base64.Encode(frame_end_ss.data, img_width*img_height);
        feature = imgbase64_begin + imgbase64_end;

    }

    return EM_SUCESS_STATE;
}


int rsv_is_samevideo(void *pvRsvHandle, std::string& videopath1,std::string& videopath2, bool &ret)
{
    TRemoveHandle *pRsvHandle = (TRemoveHandle*)pvRsvHandle;
    float fsimthresh = pRsvHandle->f32similar_thresh;
    ZBase64 base64 = pRsvHandle->base64;

    std::string feature1, feature2;
    int ret1, ret2;
    ret1 = rsv_get_video_feature(pvRsvHandle, videopath1, feature1);
    ret2 = rsv_get_video_feature(pvRsvHandle, videopath2, feature2);

    if(ret1 == EM_SUCESS_STATE && ret2 == EM_SUCESS_STATE)
    {
        std::string sub1, sub2;
        int lens = feature1.size();
        sub1 = feature1.substr(0, lens/2);
        sub2 = feature1.substr(lens/2, lens/2);

        int OutByte = 0;
        int img_width = pRsvHandle->img_width;
        int real_height = pRsvHandle->img_height - pRsvHandle->crop_height;

        string back1 = base64.Decode(sub1.c_str(), sub1.length(), OutByte);
        cv::Mat image1(img_width, real_height, CV_8U);
        image1.data = (uchar*)back1.c_str();

        string back2 = base64.Decode(sub2.c_str(), sub2.length(), OutByte);
        cv::Mat image2(img_width, real_height, CV_8U);
        image2.data = (uchar*)back2.c_str();

        std::string sub11, sub22;
        int lens2 = feature2.size();
        sub11 = feature2.substr(0, lens2/2);
        sub22 = feature2.substr(lens2/2, lens2/2);

        string back11 = base64.Decode(sub11.c_str(), sub11.length(), OutByte);
        cv::Mat image11(img_width, real_height, CV_8U);
        image11.data = (uchar*)back11.c_str();

        string back22 = base64.Decode(sub22.c_str(), sub22.length(), OutByte);
        cv::Mat image22(img_width, real_height, CV_8U);
        image22.data = (uchar*)back22.c_str();

        float fsimilar1 = 0, fsimilar2 = 0;
        fsimilar1 = compute_similarity(image1, image11);
        fsimilar2 = compute_similarity(image2, image22);

        //std::cout << "fsimilarity: " << fsimilar1 << "  " << fsimilar2 << std::endl;

        if(fsimilar1 > fsimthresh || fsimilar2 > fsimthresh)
        {
            ret = true;
        }
        else
        {
            //std::cout << "__________________________________________________________" << std::endl;
            ret = false;
        }

        return EM_SUCESS_STATE;
    }
    else
    {
        return EM_ERROR_INPUT;
    }
}


