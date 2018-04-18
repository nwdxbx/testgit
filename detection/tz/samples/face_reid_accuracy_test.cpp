//
// Created by xulishuang on 17-11-27.
//
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "../include/tuzhenfeature.h"
#include "../include/tuzhenalginterface.hpp"
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
#include "../source/reIDfeature/face_re_id.hpp"
#include <fstream>

void feature_cal_file(std::string folder,std::string file,EFeature_Cal_Type type,std::map<std::string, std::vector<float>> &feature,std::map<std::string,std::pair<string,string>>& face_pair_map)
{

    std::string file_abs_path = folder+file;
    cv::Mat frame=cv::imread(file_abs_path);
    std::vector<float> face_attribute;
    Face_Re_ID::ins().process(frame,face_attribute);

    feature[file_abs_path] = face_attribute;
    string subject = "";
    int last_slash = file_abs_path.find_last_of("_");
    //backslashIndex = pathname.find_last_of('//');

    string path = file_abs_path.substr(0,last_slash);
    int subject_slash = path.find_last_of("/");
    subject = path.substr(subject_slash+1,path.length());

    std::string one_or_two =file_abs_path.substr(last_slash+1,file_abs_path.length());
    if(one_or_two == "1.jpg"){
        face_pair_map[subject].first= file_abs_path;
    }else{
        face_pair_map[subject].second = file_abs_path;
    }
    //features[folder+file] = feature;
    //std::cout << "--file-----:"<<folder+file << "----feature-----" <<feature;
}

void feature_cal_dir(std::string dir, int depth,EFeature_Cal_Type type,std::map<std::string, std::vector<float>> &features,std::map<std::string,std::pair<string,string>>& face_pair_map)   // 定义目录扫描函数
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
            //feature_cal_dir(dir+entry->d_name+"/", depth+1,type,features);              // 递归调用自身，扫描下一级目录的内容
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
            printf("%*s%s\n", depth, "@@@@@@@@@@@@", entry->d_name);  // 输出属性不是目录的成员
            feature_cal_file(dir,entry->d_name,type,features,face_pair_map);
        }
    }
    chdir("..");                                                  // 回到上级目录
    closedir(dp);                                                 // 关闭子目录流
}

void load_face_pair_cal_face_attr(std::string path,std::map<std::string,std::vector<float>>&face_attr,std::map<std::string,std::pair<string,string>>& face_pair_map){
    feature_cal_dir(path,0,EM_FEATURE_CAL_FACE,face_attr,face_pair_map);
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

int greater_than(vector<pair<string, float> >&feature_similarity,float thresh){
    int count = 0;
    for(int i = 0; i < feature_similarity.size();i++){
        if(feature_similarity[i].second >= thresh){
            count++;
        }
        else{

        }
    }
    return  count;
}

int less_than(vector<pair<string, float> >&feature_similarity,float thresh){
    int count = 0;
    for(int i = feature_similarity.size()-1; i >=0;i--){
        if(feature_similarity[i].second <= thresh){
            count++;
        }
        else{

        }
    }
    return  count;
}

using namespace std;

template <typename T1, typename T2>
struct great_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second > b.second;
    }
};


void face_reid_accuracy_test()
{
    Face_Re_ID::ins();
    std::string same_face_pair_path="/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/face_accuracy_test/same/";
    std::string diff_face_pair_path="/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/face_accuracy_test/diff/";
    std::map<std::string,std::pair<string,string>> same_face_pair_map;
    std::map<std::string,std::pair<string,string>> diff_face_pair_map;
    std::map<std::string,float>same_face_pair_verification_val;
    std::map<std::string,float>diff_face_pair_verification_val;
    std::map<std::string,std::vector<float>>face_attr;

    load_face_pair_cal_face_attr(same_face_pair_path,face_attr,same_face_pair_map);
    load_face_pair_cal_face_attr(diff_face_pair_path,face_attr,diff_face_pair_map);

    std::cout << "feature calculation finished..."<<endl;
    std::map<std::string,std::pair<string,string>>::iterator it;
    for(it = same_face_pair_map.begin(); it != same_face_pair_map.end(); it++){
        std::string person_1 = (*it).second.first;
        std::string person_2 = (*it).second.second;
        std::string person = (*it).first;
        std::vector<float> person_1_val = face_attr[person_1];
        std::vector<float> person_2_val = face_attr[person_2];
        float simi= similarity(person_1_val,person_2_val);
        same_face_pair_verification_val[person] = simi;
    }

    for(it = diff_face_pair_map.begin(); it != diff_face_pair_map.end(); it++){
        std::string person_1 = (*it).second.first;
        std::string person_2 = (*it).second.second;
        std::string person = (*it).first;
        std::vector<float> person_1_val = face_attr[person_1];
        std::vector<float> person_2_val = face_attr[person_2];
        float simi = similarity(person_1_val,person_2_val);
        diff_face_pair_verification_val[person] = simi;
    }

    std::cout << "feature pair similarity calculation finished..."<<endl;

    vector<pair<string, float> > same_face_pair_val_sorted(same_face_pair_verification_val.begin(), same_face_pair_verification_val.end());
    sort(same_face_pair_val_sorted.begin(), same_face_pair_val_sorted.end(), great_second<string, float>());


    vector<pair<string, float> > diff_face_pair_val_sorted(diff_face_pair_verification_val.begin(), diff_face_pair_verification_val.end());
    sort(diff_face_pair_val_sorted.begin(), diff_face_pair_val_sorted.end(), great_second<string, float>());

    std::cout << "feature pair similarity sort finished..."<<endl;

    std::cout << "same_face_pair_val_sorted..."<<endl;

    for(int i = 0; i < same_face_pair_val_sorted.size();i++){
        std::cout << same_face_pair_val_sorted[i].second << endl;
    }

    std::cout << "diff_face_pair_val_sorted..."<<endl;
    for(int j = 0; j < diff_face_pair_val_sorted.size();j++){
        std::cout << diff_face_pair_val_sorted[j].second << endl;
    }

    float accuracy_max = 0.0,optimal_thresh=0;
    for(int i =0; i< 1000;i++){
        float thresh_hold = (i+1)*1.0/1000;
        int positive_count = greater_than(same_face_pair_val_sorted,thresh_hold);
        int negative_count = less_than(diff_face_pair_val_sorted,thresh_hold);
        float accuracy= (positive_count+negative_count)*1.0/(same_face_pair_val_sorted.size()+diff_face_pair_val_sorted.size());
        if(accuracy_max < accuracy) {
            accuracy_max = accuracy;
            optimal_thresh = thresh_hold;
            std::cout<< "positive count " << positive_count <<endl;
            std::cout << "negative count" << negative_count <<endl;
            std::cout << "thresh" <<"---------------"<<thresh_hold <<endl;
        }else{
            //std::cout << "accuray is ----------------------" << accuracy_max << "thresh is-----------" << thresh_hold-1.0/1000;
            //break;
            std::cout<< "positive count " << positive_count <<endl;
            std::cout << "negative count" << negative_count <<endl;
            std::cout << "accuracy" << accuracy;
            std::cout << "thresh" <<"---------------"<<thresh_hold <<endl;
        }
    }
    std::cout << "accuray is ----------------------" << accuracy_max << "thresh is-----------" << optimal_thresh;
}

int subject_index = 0;

void read_face_dataset(std::string file, std::map<std::string, std::vector<std::string>>& face_dataset,std::map<std::string,int>&subject_index_map)
{

    char buffer[256];
    ifstream in(file);
    if (! in.is_open())
    { cout << "Error opening file"; exit (1); }
    while (!in.eof() )
    {
        in.getline (buffer,255);
        cout << buffer << endl;
        std::string face_item = buffer;
        string subject = "";
        int last_slash = face_item.find_last_of(" ");

        string path = face_item.substr(0,last_slash);
        subject = face_item.substr(last_slash+1,face_item.length());
        face_dataset[subject].push_back(path);

        if(subject_index_map.count(subject) == 0){
            subject_index_map[subject] = subject_index;
            subject_index++;
        }
        std::cout << subject << ":------" <<path <<endl;
    }
}

class finder
{
public:
    finder(const int index) :index_(index){}
    bool operator ()(const std::map<std::string, int>::value_type &item)
    {
        return item.second == index_;
    }
private:
    const int index_;
};

std::pair<int,int>make_choice(int i,int j,int loop){
    srand((unsigned)time(NULL));
    int iter = 0,m;
    loop=loop >50?loop%10:loop;
    while(iter <= loop){ m = rand() % j;iter++;}
    while(true){
        int n = rand() %j;
        if(m != n){
            return std::pair<int,int>(m,n);
        }
    }

};


int make_choice(int i,int loop){
    srand((unsigned)time(NULL));
    int iter = 0,m;
    loop=loop >1000?loop%1000:loop;
    while(iter <= loop){ m = rand() % i;iter++;}
    return m;
}

float face_pair_verification(std::string file1,std::string file2){
    std::cout << file1 << ":-------------------------------" <<endl;
    std::cout << file2 << ":-------------------------------" <<endl;
    cv::Mat frame=cv::imread(file1);
    cv::Mat frame2=cv::imread(file2);
    std::vector<float> face_attribute_1,face_attribute_2;
    Face_Re_ID::ins().process(frame,face_attribute_1);
    Face_Re_ID::ins().process(frame2,face_attribute_2);
    float simi = similarity(face_attribute_1,face_attribute_2);
    return simi;
}


int gt_count(std::vector<float>&verification_val,float thresh){
    int count = 0;
    for(int i = 0; i < verification_val.size();i++){
        if(verification_val[i] >= thresh){
            count++;
        }
    }
    return count;
}


int lt_count(std::vector<float>&verification_val,float thresh){
    int count = 0;
    for(int i = 0; i < verification_val.size();i++){
        if(verification_val[i] < thresh){
            count++;
        }
    }
    return count;
}



void face_reid_accuracy_testv2()
{
    Face_Re_ID::ins();
    std::map<std::string, std::vector<std::string>> face_dataset;
    std::map<std::string,int>subject_index_map;
    std::string face_data_txt="";
    std::vector<std::pair<std::string,std::string>>same_person_set;
    std::vector<float>same_person_verification_val;
    std::vector<std::pair<std::string,std::string>>diff_person_set;
    std::vector<float>diff_person_verification_val;
    std::vector<int>random_index;
    read_face_dataset("/work/dataset/face/train.txt",face_dataset,subject_index_map);
    for(int i = 0; i < subject_index_map.size()-1;i++){
        random_index.push_back(i);
    }
    random_shuffle(random_index.begin(), random_index.end());
    std::map<std::string, int>::iterator it ;
    for(it = subject_index_map.begin(); it != subject_index_map.end();it++ )
    {
        std::cout << "subject" << (*it).first << "------" << (*it).second << endl;
    }
    std::cout << "subject size" << subject_index_map.size()<< endl;
    //std::map<std::string, int>::iterator it = subject_index_map.begin();
    for(int i = 0; i <10000;i++)
    {
        //it += random_index[i];
        string subject;
        std::cout << "random_index[i]" << random_index[i]<<endl;
        int subject_index = i%random_index.size();
        auto it = std::find_if(subject_index_map.begin(),subject_index_map.end(), finder(random_index[subject_index]));
        //if (it != subject_index_map.end())
        {
            subject = (*it).first;
        }

        std::pair<int,int> choose_pair = make_choice(0,face_dataset[subject].size(),i);
        std::cout << "subject" << subject << endl;
        std::cout << "choose_pair.first-----" << choose_pair.first<<endl;
        std::cout << "choose_pair.second----" << choose_pair.second<<endl;
        std::cout << i <<endl;
        std::cout << "face_dataset size" << face_dataset.size()<<endl;


        same_person_set.push_back(std::pair<std::string,std::string>(face_dataset[subject][choose_pair.first],face_dataset[subject][choose_pair.second]));
    }


    for(int i = 0; i < 10000;i++)
    {
        //it += random_index[i];
        string subject1,subject2;
        std::pair<int,int>choose_diff_subjects = make_choice(0,random_index.size(),i);

        //std::map<std::string, int>::iterator it1 = subject_index_map.begin();
        auto it = std::find_if(subject_index_map.begin(),subject_index_map.end(), finder(choose_diff_subjects.first));
        if (it != subject_index_map.end())
        {
            subject1 = (*it).first;
        }

        //std::map<std::string, int>::iterator it2 = subject_index_map.begin();
        auto it2 = std::find_if(subject_index_map.begin(),subject_index_map.end(), finder(choose_diff_subjects.second));
        if (it2 != subject_index_map.end())
        {
            subject2 = (*it2).first;
        }

        int subject_index = make_choice(face_dataset[subject1].size(),i);
        int subject2_index = make_choice(face_dataset[subject2].size(),i);
        //std::pair<int,int> choose_pair = make_choice(0,face_dataset[subject].size());
        diff_person_set.push_back(std::pair<std::string,std::string>(face_dataset[subject1][subject_index],face_dataset[subject2][subject2_index]));
    }

    for(int i = 0; i < same_person_set.size();i++){
        float verification = face_pair_verification(same_person_set[i].first,same_person_set[i].second);
        same_person_verification_val.push_back(verification);
    }

    for(int i = 0; i < diff_person_set.size();i++){
        float verification =face_pair_verification(diff_person_set[i].first,diff_person_set[i].second);
        diff_person_verification_val.push_back(verification);
    }

    std::sort(same_person_verification_val.begin(),same_person_verification_val.end());
    std::sort(diff_person_verification_val.begin(),diff_person_verification_val.end());


    std::cout << "same_face_pair_val_sorted..."<<endl;

    for(int i = 0; i < same_person_verification_val.size();i++){
        std::cout << same_person_verification_val[i] << endl;
    }

    std::cout << "diff_face_pair_val_sorted..."<<endl;
    for(int j = 0; j < diff_person_verification_val.size();j++){
        std::cout << diff_person_verification_val[j] << endl;
    }


    float accuracy_max = 0.0,optimal_thresh=0;
    for(int i =0; i< 1000;i++){
        float thresh_hold = (i+1)*1.0/1000;
        int positive_count = gt_count(same_person_verification_val,thresh_hold);
        int negative_count = lt_count(diff_person_verification_val,thresh_hold);
        float accuracy= (positive_count+negative_count)*1.0/(same_person_verification_val.size()+diff_person_verification_val.size());
        if(accuracy_max < accuracy) {
            accuracy_max = accuracy;
            optimal_thresh = thresh_hold;
            std::cout<< "positive count " << positive_count <<endl;
            std::cout << "negative count" << negative_count <<endl;
            std::cout << "thresh" <<"---------------"<<thresh_hold <<endl;
        }else{
            //std::cout << "accuray is ----------------------" << accuracy_max << "thresh is-----------" << thresh_hold-1.0/1000;
            //break;
            std::cout<< "positive count " << positive_count <<endl;
            std::cout << "negative count" << negative_count <<endl;
            std::cout << "accuracy" << accuracy;
            std::cout << "thresh" <<"---------------"<<thresh_hold <<endl;
        }
    }
    std::cout << "accuray is ----------------------" << accuracy_max << "thresh is-----------" << optimal_thresh;
}

int main(void)
{

    //select which function to test
    //int sltfun = 0;
//    std::cout << "Tuzhen Feature Cal Test Begin............................" << std::endl;
//    while(true);
    //person_reid_test();
    boost::thread thread1 = boost::thread(&face_reid_accuracy_testv2);
    thread1.join();
//    std::vector<int>random_index;
//    //read_face_dataset("/media/xulishuang/8117a394-1c6c-4d39-a331-f9a037392055/data/face/train.txt",face_dataset,subject_index_map);
//    for(int i = 0; i < 3000;i++){
//        random_index.push_back(i);
//    }
//    random_shuffle(random_index.begin(), random_index.end());
    return 0;
}