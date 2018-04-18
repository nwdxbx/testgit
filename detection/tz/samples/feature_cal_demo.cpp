#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "../include/tuzhenfeature.h"
#include "../include/tuzhenalginterface.hpp"
#include "../source/compare/obj_reid_feat.prototxt.pb.h"
#include <boost/thread.hpp>
#include <fcntl.h>             // 提供open()函数
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>            // 提供目录流操作函数
#include <string.h>
#include <sys/stat.h>        // 提供属性操作函数
#include <sys/types.h>         // 提供mode_t 类型
#include <stdlib.h>
#include <signal.h>
#include "../source/pedestrain_analysis/PedestrainAnalysis.h"

void feature_cal_file(std::string folder,std::string file,EFeature_Cal_Type type,std::map<std::string, std::string> &features)
{
    cv::Mat frame=cv::imread(file);
    std::string feature;
    feat_cal(frame,type,feature);
    features[folder+file] = feature;
    //std::cout << "--file-----:"<<folder+file << "----feature-----" <<feature;
}

void feature_cal_dir(std::string dir, int depth,EFeature_Cal_Type type,std::map<std::string, std::string> &features)   // 定义目录扫描函数
{
    DIR *dp;                      // 定义子目录流指针
    struct dirent *entry;         // 定义dirent结构指针保存后续目录
    struct stat statbuf;          // 定义statbuf结构保存文件属性
    if((dp = opendir(dir.c_str())) == NULL) // 打开目录，获取子目录流指针，判断操作是否成功
    {
        //printf("----------------------%s------------------------/n",dir);
        std::cout << "-------------------------dir----------" << dir;
        puts("can't open dir.");
        return;
    }
    chdir (dir.c_str());                     // 切换到当前目录
    while((entry = readdir(dp)) != NULL)  // 获取下一级目录信息，如果未否则循环
    {
        lstat(entry->d_name, &statbuf); // 获取下一级成员属性
        if(S_IFDIR &statbuf.st_mode)    // 判断下一级成员是否是目录
        {
            if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0)
                continue;

            //printf("%*s%s/\n", depth, "", entry->d_name);  // 输出目录名称
            feature_cal_dir(dir+entry->d_name+"/", depth+1,type,features);              // 递归调用自身，扫描下一级目录的内容
        }
        else {
//            char buf[1024] = {'\0'};
//            char file_path[PATH_MAX] = {'0'}; // PATH_MAX in limits.h
//            snprintf(buf, sizeof (buf), "/proc/self/fd/%d", statbuf.);
//            if (readlink(buf, file_path, sizeof(file_path) - 1) != -1) {
//                //return std::string (file_path);
//                printf("%*s%s%s\n", depth, "@@@@@@@@@@@@", entry->d_name,file_path);  // 输出属性不是目录的成员
//                feature_cal_file(entry->d_name,type,features);
//            }
            printf("%*s%s%s\n", depth, "@@@@@@@@@@@@", entry->d_name,dir);  // 输出属性不是目录的成员
            feature_cal_file(dir,entry->d_name,type,features);
        }
    }
    chdir("..");                                                  // 回到上级目录
    closedir(dp);                                                 // 关闭子目录流
}

void load_feature_database(std::string folder,EFeature_Cal_Type type,std::map<std::string, std::string> &features){
    feature_cal_dir(folder,0,type,features);
}

using namespace std;
template <typename T1, typename T2>
struct great_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second > b.second;
    }
};

void show_person_match_result(string find_object,vector<pair<string, float> >&match_results){

    string subject = "";
    int last_slash = find_object.find_last_of("/");
    //backslashIndex = pathname.find_last_of('//');

//路径名是最后一个'/'之前的字符
    string path = find_object.substr(0,last_slash);
    int subject_slash = path.find_last_of("/");
    subject = path.substr(subject_slash+1,path.length());

    int pic_width = 1600,pic_height = 900;
    int width = pic_width/10;int height = pic_height/10;
    cv::Mat result = cv::Mat(pic_height,pic_width,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat frame = cv::imread(find_object);
    cv::Mat resized;
    float factor_x = width*1.0/frame.cols;
    float factor_y = height*1.0/frame.rows;
    int width_real,height_real;
    float scale;
//    if(width >= frame.cols && height >=frame.rows){//图像小，应该放大;
//        scale = min(factor_x,factor_y);
//        width_real = frame.cols * scale;
//        height_real = frame.rows*scale;
//    } else{//要缩小
//        scale = min(factor_x,factor_y);
//        width_real = frame.cols * scale;
//        height_real = frame.rows*scale;
//    }
    scale = min(factor_x,factor_y);
    width_real = frame.cols * scale;
    height_real = frame.rows*scale;
    cv::resize(frame, resized, cv::Size(),scale,scale);
    //cvSetImageROI(result,cv::Rect(0,0,width,height));
    cv::Rect rt(width*9,height*9,width_real,height_real);
    //cout << "-----------"<<rt.x << "---" << rt.y <<"----" << rt.width << "--" <<rt.height << endl;
    //cout << "-----------"<<resized.cols << "-------------" <<resized.rows;
    cout << "------------subject------------" << subject;
    if(width_real < width){rt.x += (width-width_real)/2;}
    if(height_real < height){rt.y += (height-height_real)/2;}
    resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
    putText(result, subject, cv::Point(width*9,height*9+10), 1, 1, CV_RGB(255, 0, 0),1);
    cv::imshow("find-image", frame);
    for(int i=0; i< match_results.size();i++)
    {

        string gallary_file = match_results[i].first;
        last_slash = gallary_file.find_last_of("/");
        string gallary_path = gallary_file.substr(0,last_slash);
        int gallary_subject_slash = gallary_path.find_last_of("/");
        string gallary_subject = gallary_path.substr(gallary_subject_slash+1,gallary_path.length());

        cv::Mat frame = cv::imread(gallary_file);
        float factor_x = width*1.0/frame.cols;
        float factor_y = height*1.0/frame.rows;
        //cv::resize(frame, resized, cv::Size(), factor_x, factor_y);
        //cv::resize(frame, resized, cv::Size(width,height));
        scale = min(factor_x,factor_y);
        width_real = frame.cols * scale;
        height_real = frame.rows*scale;
        cv::resize(frame, resized, cv::Size(),scale,scale);

        //cv::Rect rt(i%10*width,i/10*height,width,height);
        cv::Rect rt(i%10*width,i/10*height,width_real,height_real);
        cout << "-----------"<<rt.x << "---" << rt.y <<"----" << rt.width << "--" <<rt.height << endl;
        cout << "-----------"<<resized.cols << "-------------" <<resized.rows;
        if(width_real < width){rt.x += (width-width_real)/2;}
        if(height_real < height){rt.y += (height-height_real)/2;}
        resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
        if(gallary_file.find(subject) != -1){
            cv::rectangle(result,cv::Rect(i%10*width,i/10*height,width,height),cv::Scalar(255,0,0),3);
        }
        putText(result, gallary_subject, cv::Point(i%10*width,i/10*height+ 10),1, 1, CV_RGB(255, 0, 0),1);
        cout << "------------gallary subject------------" << gallary_subject;
        //resized.copyTo(result(rt));
        //resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
    }
    cv::imshow("result", result);
    cv::waitKey(0);
}

void show_face_match_result(string find_object,vector<pair<string, float> >&match_results){

    string subject = "";
    int last_slash = find_object.find_last_of("_");
    //backslashIndex = pathname.find_last_of('//');

    string path = find_object.substr(0,last_slash);
    int subject_slash = path.find_last_of("/");
    subject = path.substr(subject_slash+1,path.length());

    //路径名是最后一个'/'之前的字符
    //subject = find_object.substr(0,last_slash);

    std::cout << "frame name:" << find_object<<endl;

    int pic_width = 1600,pic_height = 900;
    int width = pic_width/10;int height = pic_height/10;
    cv::Mat result = cv::Mat(pic_height,pic_width,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat frame = cv::imread(find_object);
    cv::Mat resized;
    float factor_x = width*1.0/frame.cols;
    float factor_y = height*1.0/frame.rows;
    int width_real,height_real;
    float scale;
//    if(width >= frame.cols && height >=frame.rows){//图像小，应该放大;
//        scale = min(factor_x,factor_y);
//        width_real = frame.cols * scale;
//        height_real = frame.rows*scale;
//    } else{//要缩小
//        scale = min(factor_x,factor_y);
//        width_real = frame.cols * scale;
//        height_real = frame.rows*scale;
//    }
    scale = min(factor_x,factor_y);
    width_real = frame.cols * scale;
    height_real = frame.rows * scale;
    std::cout << "frame width:" << frame.cols << " height:" << frame.rows<<endl;
    std::cout << "frame sacle:" << scale<<endl;
    cv::resize(frame, resized, cv::Size(),scale,scale);

    //cvSetImageROI(result,cv::Rect(0,0,width,height));
    cv::Rect rt(width*9,height*9,width_real,height_real);
    cout << "rt.x:"<<rt.x << " rt.y:" << rt.y <<" rt.width:" << rt.width << " rt.height:" <<rt.height << endl;
    cout << "resized.cols:"<<resized.cols << " resized.rows:" <<resized.rows<<endl;
    cout << "subject name:" << subject << " width--real:" << width_real << " height_real:" << height_real<<endl;
    cout<< "width:"<< width << " width_real:"<<width_real<<" height:"<<height<<" height_real:"<<height_real<<endl;
    if(width_real < width){rt.x += (width-width_real)/2;}
    if(height_real < height){rt.y += (height-height_real)/2;}
    cout << "copy to rt.x:" << rt.x << " rt.y:" <<rt.y << " resized.cols:" <<resized.cols << " resized.rows:" << resized.rows<<endl;
    cout << "channels count:" <<resized.channels()<<endl;
    resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
    putText(result, subject, cv::Point(width*9,height*9+10), 1, 1, CV_RGB(255, 0, 0),1);
    cv::imshow("find-image", frame);

    for(int i=0; i< match_results.size();i++)
    //for(int i=0; i< 30;i++)
    {
        string gallary_file = match_results[i].first;
        int last_slash = gallary_file.find_last_of("_");
        //backslashIndex = pathname.find_last_of('//');
        //路径名是最后一个'/'之前的字符
        path = gallary_file.substr(0,last_slash);
        subject_slash = path.find_last_of("/");
        string gallary_subject = path.substr(subject_slash+1,path.length());
        //string gallary_subject = path.substr(0,last_slash);
        //cout << "gallary file:" << gallary_file<<endl;
        cv::Mat frame = cv::imread(gallary_file);
        float factor_x = width*1.0/frame.cols;
        float factor_y = height*1.0/frame.rows;
        //cv::resize(frame, resized, cv::Size(), factor_x, factor_y);
        //cv::resize(frame, resized, cv::Size(width,height));
        scale = min(factor_x,factor_y);
        width_real = frame.cols * scale;
        height_real = frame.rows*scale;

        cv::resize(frame, resized, cv::Size(),scale,scale);

        //cv::Rect rt(i%10*width,i/10*height,width,height);
        cv::Rect rt(i%10*width,i/10*height,width_real,height_real);
        //cout << "-----------"<<rt.x << "---" << rt.y <<"----" << rt.width << "--" <<rt.height << endl;
        //cout << "-----------"<<resized.cols << "-------------" <<resized.rows<<endl;
        if(width_real < width){rt.x += (width-width_real)/2;}
        if(height_real < height){rt.y += (height-height_real)/2;}
        resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
        if(gallary_file.find(subject) != -1){
            cv::rectangle(result,cv::Rect(i%10*width,i/10*height,width,height),cv::Scalar(255,0,0),3);
        }
        putText(result, gallary_subject, cv::Point(i%10*width,i/10*height+ 10),1, 1, CV_RGB(255, 0, 0),1);
        //cout << "------------gallary subject------------" << gallary_subject<<endl;
        //resized.copyTo(result(rt));
        //resized.copyTo(result(cv::Rect(rt.x,rt.y,resized.cols,resized.rows)));
    }
    cv::imshow("result", result);
    cv::waitKey(0);
}

void person_reid_test()
{
    PedestrainAnalysis::ins();
    void *pvFeaCmpHandle = NULL;
    PedestrainAnalysis::ins();

    //get para from configure file

    feacompare_open(&pvFeaCmpHandle,0,100,EM_FEATURE_CAL_PERSON,EM_FEATURE_CMP_MULTI_CPU);
//for person test.
    //std::string folder = "./sensereid/gallery/",
            //file="./sensereid/probe/00193/00193_00001.jpg";


    char root_path[512];
    getcwd(root_path, sizeof(root_path));
    std::string folder = string(root_path)+"/data/personreid/test/"; //zhuyi test houmian de '/'
    std::string file = string(root_path)+"/data/personreid/00122_00001_p.jpg";
    std::map<std::string, std::string> features_db;

    load_feature_database(folder,EM_FEATURE_CAL_PERSON,features_db);
    std::vector<std::string> features_vec;
    std::vector<std::string> uuids_vec;
    for (auto &i : features_db)
    {
        features_vec.push_back(i.second);
        uuids_vec.push_back(i.first);
    }
    feacompare_addfeat(pvFeaCmpHandle,features_vec,uuids_vec);
    cv::Mat frame=cv::imread(file);
    std::string feature;
    feat_cal(frame,EFeature_Cal_Type::EM_FEATURE_CAL_PERSON,feature);
    std::cout << "------------------------------------------------" << feature;
    std::vector<float> scores;
    std::vector<std::string>uuids;
    if (cv::waitKey(1) == 27) {
        raise(SIGINT);
    }

    int64 t1 = cvGetTickCount();
    feacompare_process(pvFeaCmpHandle,feature,scores,uuids,0.5);
    int64 t2 = cvGetTickCount();
    cout << "feacompare_process time cost:" << (t2-t1)/cvGetTickFrequency() / 1000000 << "s" << endl;

    std::map<std::string, float> scores_map;
    for(int i = 0; i < scores.size();i++){
        scores_map[uuids[i]] = scores[i];
    }

    //map<string, int> mymap;
    vector<pair<string, float> > mapcopy(scores_map.begin(), scores_map.end());
    sort(mapcopy.begin(), mapcopy.end(), great_second<string, float>());

    cout <<"org pic:"<<file<<endl;
    for(int i = 0; i < scores.size();i++)
    {
        cout << mapcopy[i].first << " score----:" << mapcopy[i].second <<endl;
    }

    show_person_match_result(file,mapcopy);
    cout << "reslut found above ------------------------";
    //std::cout << "index: "<< maxidx << "    score: " << fmax << std::endl;
    feacompare_close(pvFeaCmpHandle);

}


void face_reid_test()
{
    PedestrainAnalysis::ins();
    void *pvFeaCmpHandle = NULL;
    //get para from configure file

//    void *pvTuzhenHandle=NULL;
//    //get para from configure file
//    TuzhenPara para;
//    para.deviceID = 0;
//    para.intervalFrames = 25;
//    //para.eoptfun = EM_OPTIONAL_ITEM_FACE_DETECT | EM_OPTIONAL_ITEM_CROWD_DETECT;
//
//    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
    feacompare_open(&pvFeaCmpHandle,0,300,EM_FEATURE_CAL_FACE,EM_FEATURE_CMP_MULTI_CPU);

//for face test.
    //std::string folder = "/media/e/tuzhen/build-tuzhen-qt5_6_2-Default/bin/facereid/test2/";
    //std::string file="/media/e/tuzhen/build-tuzhen-qt5_6_2-Default/bin/facereid/0307389_1979.jpg";

    char root_path[512];
    getcwd(root_path, sizeof(root_path));
    std::string folder = string(root_path)+"/data/facereid/test/"; //zhuyi test houmian de '/'
    std::string file = string(root_path)+"/data/facereid/0307389_1979.jpg";
    std::map<std::string, std::string> features_db;

    load_feature_database(folder,EM_FEATURE_CAL_FACE,features_db);
    std::vector<std::string> features_vec;
    std::vector<std::string> uuids_vec;
    for (auto &i : features_db)
    {
        features_vec.push_back(i.second);
        uuids_vec.push_back(i.first);
    }

    feacompare_addfeat(pvFeaCmpHandle,features_vec,uuids_vec);
    cv::Mat frame=cv::imread(file);
    std::string feature;
    feat_cal(frame,EFeature_Cal_Type::EM_FEATURE_CAL_FACE,feature);
    //std::cout << "------------------------------------------------" << feature;

    std::vector<float> scores;
    std::vector<std::string>uuids;
    if (cv::waitKey(1) == 27) {
        raise(SIGINT);
    }

    int64 t1 = cvGetTickCount();
    feacompare_process(pvFeaCmpHandle,feature,scores,uuids,0.7);
    int64 t2 = cvGetTickCount();
    cout << "feacompare_process time cost:" << (t2-t1)/cvGetTickFrequency() / 1000000 << "s" << endl;

    std::map<std::string, float> scores_map;
    for(int i = 0; i < scores.size();i++){
    //for(int i = 0; i < 30;i++){
        scores_map[uuids[i]] = scores[i];
        //cout<<"["<<i<<"]="<<scores[i]<<endl;
    }
    //map<string, int> mymap;
    vector<pair<string, float> > mapcopy(scores_map.begin(), scores_map.end());
    sort(mapcopy.begin(), mapcopy.end(), great_second<string, float>());
    cout <<"org pic:"<<file<<endl;

    for(int i = 0; i < scores.size();i++)
    //for(int i = 0; i < 30;i++)
    {
        cout << mapcopy[i].first << " score::" << mapcopy[i].second <<endl;
    }

    show_face_match_result(file,mapcopy);
    cout << "reslut found above ------------------------";
    //std::cout << "index: "<< maxidx << "    score: " << fmax << std::endl;
    feacompare_close(pvFeaCmpHandle);
}


void faceAttributeDemo()
{
    string file="/media/e/tuzhen/build-tuzhen-qt5_6_2-Default/bin/facereid/0307389_1979.jpg";
    cv::Mat frame = cv::imread(file);
    PersonInfo personinfo;
    LOG(INFO) << personinfo.coat_color << "\n";
    LOG(INFO) << personinfo.trouser_color << "\n";
    LOG(INFO) << personinfo.pos_type << "\n";
    LOG(INFO) << personinfo.backpack << "\n";
    LOG(INFO) << personinfo.sleeve_length << "\n";
    LOG(INFO) << personinfo.pants_length << "\n";
    LOG(INFO) << personinfo.head_type << "\n";
    LOG(INFO) << personinfo.gender_info << "\n";
    LOG(INFO) << personinfo.age_info << "\n";
    LOG(INFO) << personinfo.age_value << "\n";
    LOG(INFO) << personinfo.glass_info << "\n";
    LOG(INFO) << personinfo.hat_info << "\n";

    face_feature_attri(frame,personinfo);

    LOG(INFO) << personinfo.bgetface;
    LOG(INFO) << personinfo.facefea;
}

float similarity(std::vector<float>&attr1,std::vector<float>&attr2)
{
    float result =0.0,numerator=0,denominator_1=0,denominator_2=0;
    for(int i = 0; i < attr1.size();i++)
    {
        numerator += attr1[i]*attr2[i];
        denominator_1 += attr1[i]*attr1[i];
        denominator_2 += attr2[i]*attr2[i];
    }
    result = numerator/(sqrt(denominator_1)*sqrt(denominator_2));
}

void face_attr_cos(std::string file,std::string file2)
{
    std::vector<float>attr1,attr2;
    cv::Mat frame=cv::imread(file);
    std::string feature;
    feat_cal(frame,EFeature_Cal_Type::EM_FEATURE_CAL_FACE,feature);
    face_feat_entry entry1,entry2;
    entry1.ParseFromString(feature);
    for (size_t i=0; i<entry1.reid_face_feat_size(); i++)
    {
        attr1.push_back(entry1.reid_face_feat(i));
    }



    std::string feature2;
    feat_cal(frame,EFeature_Cal_Type::EM_FEATURE_CAL_FACE,feature2);
    entry2.ParseFromString(feature2);
    for (size_t i=0; i<entry2.reid_face_feat_size(); i++)
    {
        attr2.push_back(entry2.reid_face_feat(i));
    }

    float simi = similarity(attr1,attr2);
    std::cout << simi << "----------------similarity--------" << file << "-----" << file2 <<endl;
}
int main(void)
{
    std::cout << "test mode....";
//    void *pvTuzhenHandle;
//    //get para from configure file
//    TuzhenPara para;
//    para.deviceID = 0;
//    para.intervalFrames = 25;
//    //para.eoptfun = EM_OPTIONAL_ITEM_FACE_DETECT | EM_OPTIONAL_ITEM_CROWD_DETECT;
//
//    int iRtn = tuzhen_open(&pvTuzhenHandle, &para);
//
//
//    faceAttributeDemo();
//
//    tuzhen_close(pvTuzhenHandle);
    std::cout << "release mode";
    //face_reid_test();
    //select which function to test
    //int sltfun = 0;
//    std::cout << "Tuzhen Feature Cal Test Begin............................" << std::endl;
//    while(true);
    //person_reid_test();
    face_reid_test();

//#ifdef NDEBUG
    //cv::rectangle(img,rects[max_score_index],cv::Scalar(255,0,0),3);
 //#endif
    return 0;
}

