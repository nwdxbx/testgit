//
// Created by root on 17-11-18.
//

#ifndef TUZHEN_V2_17_COMMON_H_H
#define TUZHEN_V2_17_COMMON_H_H


#include <opencv2/opencv.hpp>
#include <string>


using namespace std;
using namespace cv;

//视频特征载入模式，CPU或者GPU
typedef enum
{
    EM_LOAD_MODE_CPU=0, //load feature to CPU
    EM_LOAD_MODE_GPU=1 //load feature to GPU
}EVideofeature_Load_Mode;

//特征类型
enum EFeature_Cal_Type
{
    EM_FEATURE_CAL_FACE=0, //face feature
    EM_FEATURE_CAL_PERSON=1 //person feature
};

//视频源类型
enum ESource_Type
{
    EM_SOURCE_IMAGE=0, //0为抓拍机图片
    EM_SOURCE_VIDEO=1 //1为视频流
};

//函数异常执行返回错误码
enum EReturn_Errcode
{
    EM_SUCESS_STATE=0,  //成功执行
    EM_ERROR_MALLOC,   //分配内存失败
    EM_ERROR_FEATURE_SIZE, //特征维度大小不一致
    EM_ERROR_NULL_POINT,   //空指针
    EM_ERROR_INPUT  //输入参数不合规范
};

//特征值比较方式
enum EFeature_Cmp_Type
{
    EM_FEATURE_CMP_GPU=0, // compare feature with Gpu
    EM_FEATURE_CMP_CPU=1,  // compare feature with Cpu
    EM_FEATURE_CMP_MULTI_CPU=2  // compare feature with Multi Cpu
};

//目标方向属性
enum EMotion_Dir
{
    EM_MOTION_DIR_UP=0,  //上
    EM_MOTION_DIR_DOWN,  //下
    EM_MOTION_DIR_LEFT,  //左
    EM_MOTION_DIR_RIGHT, //右
    EM_MOTION_DIR_LEFTTOP, //左上
    EM_MOTION_DIR_LEFTBTM, //左下
    EM_MOTION_DIR_RIGHTTOP, //右上
    EM_MOTION_DIR_RIGHTBTM, //右下
    EM_MOTION_DIR_STATIC,   //静止
    EM_MOTION_DIR_UNKNOWN=-1 //未知
};

////行人衣服颜色属性
//typedef enum
//{
//    EM_PERSON_COLOR_BLACK=0,  //黑色
//    EM_PERSON_COLOR_WHITE=1,  //白色
//    EM_PERSON_COLOR_GRAY=2,  //灰色
//    EM_PERSON_COLOR_RED=3,  //红色
//    EM_PERSON_COLOR_PINK=4, //粉红色
//    EM_PERSON_COLOR_BROWN=5, //棕色
//    EM_PERSON_COLOR_ORANGE=6, //橙色
//    EM_PERSON_COLOR_YELLOW=7, //黄色
//    EM_PERSON_COLOR_GREEN=8, //绿色
//    EM_PERSON_COLOR_CYAN=9, //青色
//    EM_PERSON_COLOR_BLUE=10, //蓝色
//    EM_PERSON_COLOR_PURPLE=11, //紫色
//    EM_PERSON_COLOR_NONE=12, //透明
//    EM_PERSON_COLOR_H_STRIPE=13, //水平条纹
//    EM_PERSON_COLOR_V_STRIPE=14, //竖直条纹
//    EM_PERSON_COLOR_GRID=15,     //格子
//    EM_PERSON_COLOR_UNKNOWN=-1 //未知
//}EPerson_Clothes_Color;

//行人衣服颜色属性
enum EPerson_Clothes_Color
{
    EM_PERSON_COLOR_UNKNOWN=-1, //未知
    EM_PERSON_COLOR_BLACK=0,  //黑色
    EM_PERSON_COLOR_WHITE=1,  //白色
    EM_PERSON_COLOR_GRAY=2,  //灰色
    EM_PERSON_COLOR_RED=3,  //红色
    EM_PERSON_COLOR_GREEN=8, //绿色
    EM_PERSON_COLOR_BLUE=10, //蓝色
    EM_PERSON_COLOR_OTHER=100 //其他
};

//行人衣裤长短属性（长短裤，长短袖）
enum EPerson_Clothes_Lenght
{
    EM_CLOTHES_LENGHT_LONG=0, //长
    EM_CLOTHES_LENGHT_SHORT,  //短
    EM_CLOTHES_LENGHT_UNKNOWN=-1 //未知
};

//行人是否背包属性
enum EBackpack_Info
{
    EM_BACKPACK_TRUE=0,  //背包
    EM_BACKPACK_FALSE,   //未背包
    EM_BACKPACK_UNKNOWN=-1 //未知
};

//行人是否拎东西属性
enum EHandcarry_Info
{
    EM_HANDCARRY_FALSE=0, //未拎东西
    EM_HANDCARRY_PHONE,  //拎手机
    EM_HANDCARRY_BAG,   //拎包
    EM_HANDCARRY_OTHERS, //拎其他物品
    EM_HANDCARRY_UNKNOWN=-1 //未知
};

//行人性别属性
enum EGender_Info
{
    EM_GENDER_FEMALE=0, //女
    EM_GENDER_MALE,     //男
    EM_GENDER_UNKNOWN=-1 //未知
};

//行人年龄属性
enum EAge_Info
{
    EM_AGE_CHILD=0, //小孩
    EM_AGE_YOUTH,   //青shao年
    EM_AGE_MIDDLE_AGE, //中qing年
    EM_AGE_OLD, 	//老年
    EM_AGE_UNKNOWN=-1 //未知
};

//行人戴眼镜属性
enum EGlass_Info{
    EM_GLASS_FALSE = 0, //未戴眼镜
    EM_GLASS_COMMON,    //戴普通眼镜
    EM_GLASS_SUNGLASS,  // 戴墨镜
    EM_GLASS_UNKNOWN = -1 //未知
};

//行人戴帽子属性
enum EHat_Info
{
    EM_HAT_FALSE=0,  //未戴帽子
    EM_HAT_TRUE,     //戴帽子
    EM_HAT_UNKNOWN=-1 //未知
};

//行人戴口罩属性
enum EMask_Info
{
    EM_MASK_FALSE=0,   //未戴口罩
    EM_MASK_TRUE,	//戴口罩
    EM_MASK_UNKNOWN=-1 //未知
};

//行人刘海属性
enum EFringe_Info
{
    EM_FRINGE_FALSE=0,   //没有刘海
    EM_FRINGE_TRUE,	//有刘海
    EM_FRINGE_UNKNOWN=-1 //未知
};

//行人光头属性
enum EBare_Info
{
    EM_BARE_FALSE=0,   //非光头
    EM_BARE_TRUE,	//光头
    EM_BARE_UNKNOWN=-1 //未知
};

//人群异常行为属性
enum EPersons_Behavior_Type
{
    EM_PERSONS_BEHAVIOR_NORMAL=0, //人群正常秩序
    EM_PERSONS_BEHAVIOR_CROWN_GATHER=1, //人群聚集
    EM_PERSONS_BEHAVIOR_FIGHT,    //人群群殴
    EM_PERSONS_BEHAVIOR_UNKNOWN=-1 //未知
};

//行人姿态属性
enum EPerson_Posture_Type
{
    EM_PERSON_POSTURE_WALK=0,  //走
    EM_PERSON_POSTURE_RUN,     //跑
    EM_PERSON_POSTURE_STILL,   //静止
    EM_PERSON_POSTURE_UNKNOWN=-1 //未知
};

//功能开启选项
//bit0： 人脸检测，bit1：人头检测，bit2：人群聚集检测
//typedef enum
//{
//    EM_OPTIONAL_ITEM_FACE_DETECT = 1,
//    EM_OPTIONAL_ITEM_HEAD_DETECT = 2,
//    EM_OPTIONAL_ITEM_CROWD_DETECT = 4
//}EOptional_Item;
enum EOptional_Item
{
    EM_OPTIONAL_ITEM_PEDESTRAIN = 1,
    EM_OPTIONAL_ITEM_VEHICLE = 2,
    EM_OPTIONAL_ITEM_HYBRID = 3,
    EM_OPTIONAL_ITEM_CROWD_ESTIMATE = 4
};


enum EPerson_Head_Type{
    EM_PERSON_HEAD_FRONT=0,
    EM_PERSON_HEAD_BACK,
    EM_PERSON_HEAD_UNKNOWN=-1
} ;

struct PersonInfo {
    int structsize;    //结构体大小
    vector<Vec3f> pts;   //人体关节点坐标
    vector<Vec2f> face_landmarks;//人脸关键点特征
    Rect personrct;      //行人目标框
    EPerson_Clothes_Color coat_color;  //上衣颜色
    float coat_color_score=0;                  //上衣颜色分数
    EPerson_Clothes_Color trouser_color; //裤子颜色
    float trouser_color_score=0;                 //裤子颜色分数
    EPerson_Posture_Type pos_type;  //姿态类型

    EBackpack_Info backpack;		//背包属性
    EPerson_Clothes_Lenght sleeve_length;  //长短袖属性
    float sleeve_length_score=0;                    //袖子分数
    EPerson_Clothes_Lenght pants_length;	//长短裤属性
    float  pants_length_score=0;                     //裤子分数
    float person_integrity;


    int  objectID;	//目标ID号
    bool breport;     //是否上报
    float fscore;     //目标得分值，用于更换新的特征
    string personfea; //行人特征值

    bool bgetface = false; //是否检测到人脸，如果没有，以下属性都无效
    EPerson_Head_Type head_type; // 人脸类型，正脸或者背脸
    float head_score=0;
    Rect facerct;  //人脸目标框
    EGender_Info gender_info; //性别
    float gender_score=0;
    EAge_Info age_info; //年龄类型
    int age_value=0;  //年龄具体值
    float age_score=0;
    EGlass_Info glass_info; //戴眼镜属性
    float glass_score=0;
    EHat_Info hat_info; //戴帽子属性
    float hat_score=0;
    EMask_Info mask_info; //戴口罩属性
    float mask_score=0;
    EBare_Info bare_info; // 光头属性
    float bare_score=0;
    EFringe_Info fringe_info; // 刘海属性
    float fringe_score=0;
    string facefea; //人脸特征值

    PersonInfo():coat_color(EM_PERSON_COLOR_UNKNOWN),
                 trouser_color(EM_PERSON_COLOR_UNKNOWN),
                 pos_type(EM_PERSON_POSTURE_UNKNOWN),
                 backpack(EM_BACKPACK_UNKNOWN),
                 sleeve_length(EM_CLOTHES_LENGHT_UNKNOWN),
                 pants_length(EM_CLOTHES_LENGHT_UNKNOWN),
                 person_integrity(0),
                 head_type(EM_PERSON_HEAD_UNKNOWN),
                 gender_info(EM_GENDER_UNKNOWN),
                 age_info(EM_AGE_UNKNOWN),
                 age_value(-1),
                 glass_info(EM_GLASS_UNKNOWN),
                 hat_info(EM_HAT_UNKNOWN),
                 mask_info(EM_MASK_UNKNOWN),
                 bare_info(EM_BARE_UNKNOWN),
                 fringe_info(EM_FRINGE_UNKNOWN)
    {


        //LOG(INFO) << "test---------construct-------------";
        //std::cout << "test---------construct-------------";
    }

};

//
enum VehicleType
{
    VehicleType_Unknown = -1,
    Car=0,
    Truck,
    Bus,
    Motobike,
    Bike
};

enum DetectionType{
    PEDESTRAIN=1,
    BICYCLE,
    CAR,
    MOTORCYCLE,
    BUS,
    TRAIN,
    TRUCK,
    BACKPACK,
    UMBRELLA,
    HANDBAG,
    CELL_PHONE,
    SUITCASE,
    TIE
};



struct VehicleInfo{
    cv::Rect vehicle_location;//车辆位置
    //std::string vehicle_id;
    VehicleType vehicle_type;//车辆类型，0,car;1,truck;2,bus;3,motobike;4,Bike
    std::string card_no;//车牌号码
    std::string card_color;//车牌颜色
    std::string vehicle_brand;//汽车品牌
    std::string vehicle_color;//汽车颜色
    bool breport;     //是否上报
    float fscore;     //目标得分值，用于更换新的特征
    int objectID;
};


struct CrowdEstimateInfo{
    cv::Mat heat_map;
    float estimated_pedestrain_count;
};




#endif //TUZHEN_V2_17_COMMON_H_H
