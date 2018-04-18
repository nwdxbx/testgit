//
// Created by xulishuang on 18-1-10.
//
#include <opencv2/opencv.hpp>
#include "../source/openpose/openposeCaffe.hpp"
#include <string>
#include <fcntl.h>             // 提供open()函数
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>            // 提供目录流操作函数
#include <string.h>
#include <sys/stat.h>        // 提供属性操作函数
#include <sys/types.h>         // 提供mode_t 类型
#include <stdlib.h>
#include <signal.h>
#include <boost/thread.hpp>
#include <fstream>


std::string test_result_dir="/work/dataset/data_generator/mask-rcnn/shenyang03/dest/keypoints_detect/";
std::string keypoints_result_dir="/work/dataset/data_generator/mask-rcnn/shenyang03/dest/keypoints_label/";
std::vector <cv::Scalar> colors_;
std::vector<std::pair<size_t, size_t>> matchpair;
//openpose_caffe::ins().getmatchpair(matchpair);
const std::vector<unsigned int> render_pairs {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17};
//std::vector<std::pair<size_t, size_t> > matchpair;

void draw_keypoints(std::string file,cv::Mat img,std::vector<std::vector<cv::Vec3f>>& pts)
{
    std::cout << pts.size();
    int ext_pos = file.find_last_of(".");
    //backslashIndex = pathname.find_last_of('//');
    //路径名是最后一个'/'之前的字符
    std::string shot_name = file.substr(0,ext_pos);
    //int subject_slash = path.find_last_of("/");
    //subject = path.substr(subject_slash+1,path.length());
    if(pts.size() == 0){
        cv::imwrite(test_result_dir+file,img);
        return;
    }
    std::ofstream out(keypoints_result_dir+shot_name+".txt");
    for (size_t i=0; i<pts.size(); i++)
    {
        auto &pts_one=pts[i];
        if (pts_one.empty())
            continue;

        for(int s = 0; s < pts_one.size();s++){
            out << pts_one[s][0] <<" "<< pts_one[s][1] << " " <<pts_one[s][2] << " ";
        }
        out << std::endl;

        auto f=[&](int idpart, const cv::Scalar &color)
        {
            std::pair<size_t, size_t> &bodypart=matchpair[idpart];
            if (std::min(pts_one[bodypart.first][2], pts_one[bodypart.second][2]) < 0.1)
                return;
            int id1=bodypart.first;
            int id2=bodypart.second;
            cv::line(img,
                     cv::Point(pts_one[id1][0], pts_one[id1][1]),
                     cv::Point(pts_one[id2][0], pts_one[id2][1]),
                     color, 2);
        };
        for (size_t idpart=0; idpart<matchpair.size(); idpart++)
        {
            f(idpart, colors_[idpart]);
        }
    }
    out.close();
    cv::imwrite(test_result_dir+file,img);

}

void keypoints_detector_file(std::string dir,std::string file)
{
    cv::Mat frame=cv::imread(file);
    std::vector<std::vector<cv::Vec3f>> pts;
    pts.clear();
    openpose_caffe::ins().process(frame, pts);

    draw_keypoints(file,frame,pts);

}
void keypoints_detector_dir(std::string dir, int depth)   // 定义目录扫描函数
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
            keypoints_detector_dir(dir+entry->d_name+"/", depth+1);              // 递归调用自身，扫描下一级目录的内容
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
            keypoints_detector_file(dir,entry->d_name);
        }
    }
    chdir("..");                                                  // 回到上级目录
    closedir(dp);                                                 // 关闭子目录流
}

void keypoints_collecting_thread(){
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    openpose_caffe::ins();
    std::string img_dir = "/work/dataset/data_generator/mask-rcnn/shenyang03/dest/JPEGImages";
    keypoints_detector_dir(img_dir,0);
}

int main(void) {
    colors_.push_back(cv::Scalar(0, 255, 255));
    colors_.push_back(cv::Scalar(0, 255, 0));
    colors_.push_back(cv::Scalar(255, 255, 0));
    colors_.push_back(cv::Scalar(255, 0, 255));
    colors_.push_back(cv::Scalar(0, 0, 255));
    colors_.push_back(cv::Scalar(255, 125, 0));
    colors_.push_back(cv::Scalar(125, 255, 0));
    colors_.push_back(cv::Scalar(0, 255, 125));
    colors_.push_back(cv::Scalar(0, 125, 255));
    colors_.push_back(cv::Scalar(255, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 255));
    colors_.push_back(cv::Scalar(125, 125, 255));
    colors_.push_back(cv::Scalar(125, 255, 125));
    colors_.push_back(cv::Scalar(255, 125, 125));
    colors_.push_back(cv::Scalar(0, 0, 125));
    colors_.push_back(cv::Scalar(125, 0, 0));
    colors_.push_back(cv::Scalar(0, 125, 0));

    for (size_t i=0; i<render_pairs.size()/2; i++)
        matchpair.push_back({render_pairs[2*i], render_pairs[2*i+1]});

    boost::thread t1(&keypoints_collecting_thread);
    t1.join();

}