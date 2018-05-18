#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "mtcnn.hpp"
#include "facedetermine.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>


std::vector<cv::Point2f> coordlandmarks;

//seeta::FaceDetection _detector("/home/ffh/work/project/caffe-face/models/seeta_fd_frontal_v1.0.bin");

template<class T>
bool calcDirect(cv::Rect faceRect, std::vector<cv::Point_<T>> landmark)
{
    //frontal or profile
    float rect_len = std::sqrt(faceRect.width*faceRect.width*1.0/4+faceRect.height*faceRect.height*1.0/4);
    float nose_xdis = abs(landmark[2].x - (faceRect.x + faceRect.width/2));
    float nose_ydis = abs(landmark[2].y - (faceRect.y + faceRect.height/2));
    float face_len = sqrt( nose_xdis*nose_xdis + nose_ydis*nose_ydis);
    float dist = face_len/rect_len;
    float dist_threshold = 0.3;
    float nose_leye_dis = landmark[2].x - landmark[0].x;
    float nose_reye_dis = landmark[1].x - landmark[2].x;
    float nose_eys_dist = std::min(nose_leye_dis, nose_reye_dis)*1.0/std::max(nose_leye_dis, nose_reye_dis);
    float nose_eys_threshold = 0.3;
    if (dist < dist_threshold && nose_eys_dist > nose_eys_threshold)
        return true;
    return false;
}

static void correct_img_roi(cv::Mat &m, cv::Rect &r)
{
    if (r.x < 0) r.x = 0;
    if (r.y < 0) r.y = 0;
    if (r.x > m.cols - 1) r.x = m.cols - 1;
    if (r.y > m.rows - 1) r.y = m.rows - 1;

    if (r.x + r.width > m.cols-1) r.width = m.cols - r.x -1;

    if (r.y + r.height > m.rows -1) r.height = m.rows - r.y -1;
}

//int detectFrontalFace()
//{
//    int dirCount[8]={0,0,0,0,0,0,0,0};
//    int frontalCount = 0;
//    _detector.SetMinFaceSize(40);
//    _detector.SetScoreThresh(0.f);
//    _detector.SetImagePyramidScaleFactor(0.85f);
//    _detector.SetWindowStep(4,4);
//    //std::ifstream profile_list("./err_img.txt");
//    std::ifstream profile_list("./frontal_all.txt");

//    std::string imgDir;
//    std::vector<cv::Mat> imgs;
//    while(getline(profile_list, imgDir))
//    {
//        int space_index = imgDir.find(" ");
//        std::string ori_imgDir = imgDir.substr(0, space_index);
//        std::cout<<ori_imgDir<<std::endl;
//        std::cout<<ori_imgDir<<std::endl;

//        cv::Mat testImg = cv::imread(ori_imgDir);
//        cv::resize(testImg, testImg,cv::Size(250,250));
//        imgs.push_back(testImg);
//    }
//    std::vector<std::vector<alg::MTCnn::FaceInf>> faceInfo;
//    alg::MTCnn::ins().Detect(imgs, faceInfo);
//    for(int i=0; i<imgs.size(); i++)
//    {
//        cv::Mat showImg = imgs[i].clone();
////        imgs.push_back(testImg);
////        std::vector<std::vector<MTCnn::FaceInf>> faceInfo;
////        mtcnn_fun.Detect(imgs, faceInfo);
////        cv::Mat showImg = testImg.clone();
////        if (faceInfo.size() < 1)
////        {
////            continue;
////        }

//        if (faceInfo[i].size() < 1)
//        {
//            continue;
//        }

//        for(int j = 0; j<faceInfo[i].size(); j++)
//        {
//            alg::MTCnn::FaceInf curInfo = faceInfo[i][j];
//            if(!curInfo.isCorrect)
//            {
//                continue;
//            }
//            std::vector<cv::Point2f> landmark;

//            landmark.push_back(curInfo.leye);
//            landmark.push_back(curInfo.reye);
//            landmark.push_back(curInfo.nose);
//            landmark.push_back(curInfo.lmouth);
//            landmark.push_back(curInfo.rmouth);
//            cv::rectangle(showImg, curInfo.faceRect,cv::Scalar(0,0,255),1);
//            for(int k =0; k< landmark.size(); k++)
//            {
//                cv::circle(showImg, landmark[k], 2, cv::Scalar(0,255,0));
//            }
//        }
//        frontalCount ++;
//        cv::imshow("showImg",showImg);
//        cv::waitKey(0);
//    }
//    std::cout<<"total count: "<<frontalCount<<std::endl;
//    return 0;
//}


int detectFace()
{
    int dirCount[8]={0,0,0,0,0,0,0,0};
    int frontalCount = 0;
    //std::ifstream profile_list("./frontal_test_list.txt");
    std::ifstream profile_list("/media/e/FrameWork/face_redress/test.txt");

    int frontalNum = 0;
    int profileNum = 0;
    int loujian = 0;
    int wujian = 0;
    std::string imgDir;
    cv::namedWindow("showImg",cv::WINDOW_NORMAL);
    while(getline(profile_list, imgDir))
    {
        int space_index = imgDir.find(" ");
        std::string ori_imgDir = imgDir.substr(0, space_index);
        std::string ori_label = imgDir.substr(space_index+1);
        int labelNUm = std::stoi(ori_label);
        std::cout<<ori_imgDir<<" "<<ori_label<<std::endl;
        if(labelNUm == 1)
            frontalNum ++;
        else
            profileNum ++;

        std::vector<cv::Point2f> landmark;
        cv::Mat srcImg = cv::imread(ori_imgDir);
        cv::Mat testImg;
        float factor = 1.0;
        testImg = srcImg.clone();
        //cv::resize(srcImg, testImg, cv::Size(), factor, factor);
        //cv::resize(srcImg, testImg, cv::Size(srcImg.cols*factor, srcImg.rows*factor), 0, 0, cv::INTER_CUBIC);
        std::vector<cv::Mat> imgs;
        imgs.push_back(testImg);
        cv::Mat showImg = testImg.clone();
        std::vector<std::vector<alg::MTCnn::FaceInf>> faceInfo;
        alg::MTCnn::ins().Detect(imgs, faceInfo);
        if (faceInfo.size() < 1)
        {
            continue;
        }
        if (faceInfo[0].size() < 1)
        {
            if(labelNUm == 0)
                frontalCount ++;
            if(labelNUm == 1)
                loujian ++;
            std::cout<<"no face"<<std::endl;
            continue;
        }
//        else if(!faceInfo[0][0].isCorrect)
//        {
//            if(labelNUm == 0)
//                frontalCount ++;
//            if(labelNUm == 1)
//                loujian ++;
//            std::cout<<"no face"<<std::endl;
//            continue;
//        }
        cv::Rect faceRect ;
        for(int k = 0; k< faceInfo[0].size(); k++)
        {
            alg::MTCnn::FaceInf curInfo = faceInfo[0][k];
            landmark.push_back(curInfo.leye);
            landmark.push_back(curInfo.reye);
            landmark.push_back(curInfo.nose);
            landmark.push_back(curInfo.lmouth);
            landmark.push_back(curInfo.rmouth);
            for(int i =0; i< landmark.size(); i++)
            {
                cv::circle(showImg, landmark[i], 2, cv::Scalar(0,255,0));
            }
            cv::rectangle(showImg, curInfo.faceRect, cv::Scalar(0,0,255), 2);
            faceRect= curInfo.faceRect;
        }
        cv::imshow("showImg",showImg);
        //cv::waitKey(0);

        if(labelNUm == 1)
            frontalCount ++;
        if(labelNUm == 0)
            wujian ++;

        cv::Mat warp_frame;
        cv::Mat oriImg = testImg(faceRect).clone();
        cv::imshow("oriImg",oriImg);


        cv::Mat warp_mat = cv::estimateRigidTransform(landmark, coordlandmarks, false);
        if (warp_mat.cols == 3 && warp_mat.rows == 2)
        {
            cv::warpAffine(testImg, warp_frame, warp_mat, cv::Size(112,112));
        }
        else
        {
            //std::cout<<"not warpaffine"<<std::endl;
            continue;
            //cv::resize(faceImg, warp_frame, cv::Size(96,112));
        }
        std::cout<<"width: "<<warp_frame.cols<<" height: "<<warp_frame.rows<<std::endl;
        cv::imshow("warp_frame",warp_frame);
        cv::waitKey(0);
    }
    cv::destroyWindow("showImg");
    for(int i=0; i< 8; i++)
    {
        std::cout<<"result: "<<dirCount[i]<<std::endl;
    }
    std::cout<<"total count: "<<frontalNum<<" "<<profileNum<<" "<<frontalCount<<std::endl;
    std::cout<<"loujian: "<<loujian<<"  wujian:"<<wujian<<std::endl;
    return 0;
}

//int cutImgFace()
//{
//    std::ifstream profile_list("./addface0716.txt");
//    _detector.SetMinFaceSize(40);
//    _detector.SetScoreThresh(0.f);
//    _detector.SetImagePyramidScaleFactor(0.85f);
//    _detector.SetWindowStep(4,4);

//    std::vector<int> img_param;
//    img_param.push_back(CV_IMWRITE_JPEG_QUALITY);
//    img_param.push_back(100);

//    std::string imgDir;
//    cv::namedWindow("showImg",cv::WINDOW_NORMAL);
//    int img_num = 0;
//    while(getline(profile_list, imgDir))
//    {
//        cv::Mat img = cv::imread(imgDir);
//        cv::Mat img_gray;
//        if (img.channels() != 1)
//          cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
//        else
//          img_gray = img;

//        seeta::ImageData img_data;
//        img_data.data = img_gray.data;
//        img_data.width = img_gray.cols;
//        img_data.height = img_gray.rows;
//        img_data.num_channels = 1;
//        std::vector<seeta::FaceInfo> faces;
//        faces = _detector.Detect(img_data);
//        cv::Rect res;
//        if (faces.size() != 0)
//        {
//            //score = faces[0].score;
//            res.x = faces[0].bbox.x;
//            res.y = faces[0].bbox.y;
//            res.width = faces[0].bbox.width;
//            res.height = faces[0].bbox.height;
//            correct_img_roi(img_gray, res);
//        }
//        else
//            continue;
//        cv::Mat saveImg = img(res).clone();
////        char savePath[256];
////        sprintf(savePath, "%simg_%d.jpg","/home/ffh/data/face_train/data/addimg0720face/",img_num);
////        std::string pathStr(savePath);
//        std::string pathStr = imgDir.replace(imgDir.find("addface0716"), 11 ,"addface0716_cut");
//        cv::imwrite(pathStr,saveImg,img_param);
//        img_num ++;
//    }
//    return 0;
//}

int testVideo()
{
    cv::VideoCapture cap("/home/ffh/work/project/face_redress/20-177.mp4");
    //cv::VideoCapture cap("rtsp://admin:admin123@192.168.103.71");
    //cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_POS_FRAMES,3000);
    char savePath[256];
    int saveNum = 0;
    cv::Mat testImg;
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    //std::ifstream ifs("/home/ffh/work/project/face_redress/imgs/tmp/tmp.txt");
    //std::ifstream ifs("/media/e/data/celaba/img_celeba-7z/img_celeba.txt");
    std::ifstream ifs("/media/e/data/celaba/img_celeba-7z/big.txt");
    std::string stringLine;

    //while(1)
    while(getline(ifs, stringLine))
    {
        //cap.set(CV_CAP_PROP_EXPOSURE,-1);
        //cap >> testImg;
        // cv::imwrite("img.jpg",img);
        std::cout<<stringLine<<std::endl;
        testImg = cv::imread(stringLine);
        if(testImg.empty()) break;
        //cv::resize(testImg, testImg, cv::Size(1280, 720));
        //testImg = cv::imread("/home/ffh/data/Video/error_img/error_img_138.jpg");
        std::vector<cv::Mat> imgs;
        imgs.push_back(testImg);
        cv::Mat showImg = testImg.clone();
        std::vector<std::vector<alg::MTCnn::FaceInf>> faceInfo;
        alg::MTCnn::ins().Detect(imgs, faceInfo);

        if (faceInfo.size() < 1)
        {
//            cv::imshow("showImg",showImg);
//            cv::waitKey(1);
            continue;
        }
        if (faceInfo[0].size() < 1)
        {
            std::cout<<"no face"<<std::endl;
//            cv::imshow("showImg",showImg);
//            cv::waitKey(1);
            continue;
        }
        for(int k = 0; k< faceInfo[0].size(); k++)
        {
            std::vector<cv::Point2f> landmark;
            alg::MTCnn::FaceInf curInfo = faceInfo[0][k];
            //if(!curInfo.isCorrect)
                //continue;
            //std::cout<<"frontalCon: "<<frontalCon<<std::endl;
            landmark.push_back(curInfo.leye);
            landmark.push_back(curInfo.reye);
            landmark.push_back(curInfo.nose);
            landmark.push_back(curInfo.lmouth);
            landmark.push_back(curInfo.rmouth);
            for(int i =0; i< landmark.size(); i++)
            {
                //cv::circle(showImg, landmark[i], 2, cv::Scalar(0,255,0));
            }
            //std::cout<<"face: "<<curInfo.faceRect.width<<"  "<<curInfo.score<<std::endl;
            cv::rectangle(showImg, curInfo.faceRect, cv::Scalar(0,0,255), 1);
            cv::Rect facRec = curInfo.faceRect;
            facRec.x = std::max(0, facRec.x);
            facRec.y = std::max(0, facRec.y);
            facRec.width = std::min(testImg.cols - facRec.x, facRec.width);
            facRec.height = std::min(testImg.rows - facRec.y, facRec.height);
            cv::Mat saveImg = testImg(facRec).clone();
            saveNum += 1;
//            sprintf(savePath,"%serror_img_%d.jpg","/home/ffh/data/Video/error_img/",saveNum);
//            std::string imgPath(savePath);
//            cv::imwrite(imgPath, testImg, compression_params);
        }
        //cv::imshow("showImg",showImg);
        for(int k = 0; k< faceInfo[0].size(); k++)
        {
            alg::MTCnn::FaceInf curInfo = faceInfo[0][k];
            //if(curInfo.isCorrect)
                //cv::waitKey(1);
        }
        //cv::waitKey(1);
    }
    return 0;
}


int save_img()
{
    int saveNum = 0;
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    std::string source_txt = "/media/e/data/celaba/img_align_celeba.txt";
    std::string save_folder = "/media/f/jiakao/imgs/mtcnn_img/celeba/align_text/";
    std::ifstream infile(source_txt.c_str());
    std::string line;

    while (std::getline(infile, line))
    {
        std::string st_2;
        st_2.insert(st_2.begin(),line.begin(),line.end());
        int index = st_2.rfind("/");
        std::string img_name = st_2.substr(index+1);
        int pt_index = img_name.find(".");
        std::string text_name = img_name.substr(0, pt_index) + ".txt";
        std::cout<<st_2<<std::endl;
        cv::Mat testImg = cv::imread(st_2);

        saveNum++;

        std::vector<cv::Mat> imgs;
        imgs.push_back(testImg);
        //cv::Mat showImg = testImg.clone();
        std::vector<std::vector<alg::MTCnn::FaceInf>> faceInfo;

        alg::MTCnn::ins().Detect(imgs, faceInfo);
        if (faceInfo.size() < 1)
        {
            continue;
        }
        if (faceInfo[0].size() < 1)
        {
            std::cout<<"no face"<<std::endl;
            continue;
        }

        std::string savePath = save_folder+text_name;
        std::ofstream out(savePath);
        if (out.is_open())
        {

            for(int k = 0; k< faceInfo[0].size(); k++)
            {
                std::vector<cv::Point2f> landmark;
                alg::MTCnn::FaceInf curInfo = faceInfo[0][k];
//                if(!curInfo.isCorrect)
//                    continue;
//                std::cout<<"frontalCon: "<<frontalCon<<std::endl;
//                landmark.push_back(curInfo.leye);
//                landmark.push_back(curInfo.reye);
//                landmark.push_back(curInfo.nose);
//                landmark.push_back(curInfo.lmouth);
//                landmark.push_back(curInfo.rmouth);
//                cv::rectangle(showImg, curInfo.faceRect, cv::Scalar(0,0,255), 1);
                cv::Rect facRec = curInfo.faceRect;

                out<<facRec.x<<" "<<facRec.y<<" "<<facRec.width<<" "<<facRec.height<<std::endl;
            }
            out.close();
        }
//        for(int k = 0; k< faceInfo[0].size(); k++)
//        {
//            alg::MTCnn::FaceInf curInfo = faceInfo[0][k];
//            if(curInfo.isCorrect)
//                cv::waitKey(1);
//        }
//        cv::waitKey(1);
    }

    return 0;
}



int mtcdetec()
{
    std::ifstream profile_list("./test.txt");
    std::string imgDir;
    double min=1000000.0;
    double max=0.0;
    double me=0.0;
    double sm = 0.0;
    int count = 0;
    while(getline(profile_list, imgDir))
    {
        cv::Mat srcImg = cv::imread(imgDir);
        if(srcImg.channels()!=3) continue;
        cv::Mat testImg;
        testImg = srcImg.clone();
        std::vector<cv::Mat> imgs;
        imgs.push_back(testImg);
        std::vector<std::vector<alg::MTCnn::FaceInf>> faceInfo;
        double t = (double)cv::getTickCount();
        alg::MTCnn::ins().Detect(imgs, faceInfo);
        t =((double)cv::getTickCount()-t) / cv::getTickFrequency();
        std::cout<<"the time is : "<<t<<std::endl;
        if(t>max) max=t;
        if(t<min) min=t;
        count++;
        sm +=t;
        float score = 0.0;
        int index = -1;
        for(int k = 0; k< faceInfo[0].size(); k++)
        {
            if(faceInfo[0][k].score>score)
            {
                score = faceInfo[0][k].score;
                index = k;
            }
        }
        cv::Rect faceRect ;
        std::vector<cv::Point2f> landmark;
        landmark.push_back(faceInfo[0][index].leye);
        landmark.push_back(faceInfo[0][index].reye);
        landmark.push_back(faceInfo[0][index].nose);
        landmark.push_back(faceInfo[0][index].lmouth);
        landmark.push_back(faceInfo[0][index].rmouth);
        faceRect = faceInfo[0][index].faceRect;

        cv::Mat warp_frame;
        cv::Mat oriImg = testImg(faceRect).clone();
//        cv::imshow("oriImg",oriImg);
        cv::Mat warp_mat = cv::estimateRigidTransform(landmark, coordlandmarks, false);
        if (warp_mat.cols == 3 && warp_mat.rows == 2)
            cv::warpAffine(testImg, warp_frame, warp_mat, cv::Size(112,112));
        else
            continue;
        cv::imshow("warp_frame",warp_frame);
        cv::waitKey(0);
    }
    me = 1.0*sm/count;
    std::cout<<"min: "<<min<<" "<<"max: "<<max<<" "<<"mean: "<<me<<" "<<"count: "<<count<<std::endl;

}

void getfaceFeature()
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/e/FrameWork/tuzhen_face/bin/models/model.prototxt",caffe::TEST));
    net_->CopyTrainedLayersFrom("/media/e/FrameWork/tuzhen_face/bin/models/model.caffemodel");

    net_->input_blobs()[0]->Reshape(2,3,112,112);
    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(127.5,127.5,127.5);

    std::ifstream flist("/media/e/bat/QT_program/tzAl/build/face.txt");
    std::string imgDir;
    double min=1000000.0;
    double max=0.0;
    double me=0.0;
    double sm = 0.0;
    int count = 0;
    while(getline(flist, imgDir))
    {
        double t = (double)cv::getTickCount();
        cv::Mat img=cv::imread(imgDir);
        if(img.empty()) continue;
        if(img.channels()!=3) continue;
        cv::Mat sample;
        std::vector<cv::Mat> sample_single;
        sample_single.resize(2);
        cv::cvtColor(img,sample,cv::COLOR_BGR2RGB);
        cv::resize(sample,sample,cv::Size(112,112));
        sample.convertTo(sample_single[0],CV_32FC3);
        sample_single[0] -=mean;
        sample_single[0] =0.0078125*sample_single[0];
        cv::flip(sample_single[0],sample_single[1],1);


        float *input_data=NULL;
        input_data=input_layer->mutable_cpu_data();
        std::vector<std::vector<cv::Mat>> input_channels;
        input_channels.resize(2);
        for(int k=0;k<2;k++)
        {
            for(int i=0;i<input_layer->channels();i++)
            {
                cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
                input_channels[k].push_back(channel);
                input_data +=(input_layer->height())*(input_layer->width());
            }
            cv::split(sample_single[k],input_channels[k]);
        }
        net_->Forward();
        const boost::shared_ptr<caffe::Blob<float> > feature_blob = net_->blob_by_name("fc1");
        const float *feature_data = feature_blob->cpu_data();
        std::vector<float> feature;
        feature.resize(feature_blob->channels());
        float sum =0.0;
        for(int i=0;i<feature_blob->channels();i++)
        {
            feature[i] = feature_data[i]+feature_data[feature_blob->channels()+i];
            sum +=feature[i];
        }
        for(int i=0;i<feature_blob->channels();i++)
            feature[i] /=sum;
        t =((double)cv::getTickCount()-t) / cv::getTickFrequency();
        if(t>max) max=t;
        if(t<min) min=t;
        count++;
        sm +=t;
//        std::cout<<"sum: "<<sm<<std::endl;
        //std::cout<<"the time is : "<<t<<std::endl;
    }
    me = 1.0*sm/count;
    std::cout<<"min: "<<min<<" "<<"max: "<<max<<" "<<"mean: "<<me<<" "<<"count: "<<count<<std::endl;
}

void dir_get_file(std::string dir,std::vector<std::string> &absfilenames)
{
    DIR *dp;
    struct dirent *entry;
    struct stat statbuf;
    if((dp=opendir(dir.c_str()))==NULL)
    {
        std::cout<<"can't open dir......"<<dir<<std::endl;
        return;
    }
    //chdir(dir.c_str());
    while ((entry=readdir(dp))!=NULL) {
       lstat(entry->d_name,&statbuf);
       if(entry->d_type & DT_DIR /*S_ISDIR(statbuf.st_mode)*/)
       {
           if(strcmp(".",entry->d_name) == 0 ||strcmp("..",entry->d_name) == 0)
               continue;
           dir_get_file(dir+entry->d_name+"/",absfilenames);
       }
       else
           absfilenames.push_back(dir+entry->d_name);
    }
    //chdir("..");
    closedir(dp);
}

void allign()
{
    std::vector<cv::Point2f> landmarks112;
    std::vector<cv::Point2f> landmarks96;

    landmarks112.push_back(cv::Point2f(38.2946, 51.6963));
    landmarks112.push_back(cv::Point2f(73.5318, 51.5014));
    landmarks112.push_back(cv::Point2f(56.0252, 71.7366));
    landmarks112.push_back(cv::Point2f(41.5493, 92.3655));
    landmarks112.push_back(cv::Point2f(70.7299, 92.2041));

    landmarks96.push_back(cv::Point2f(30.2946, 51.6963));
    landmarks96.push_back(cv::Point2f(65.5318, 51.5014));
    landmarks96.push_back(cv::Point2f(48.0252, 71.7366));
    landmarks96.push_back(cv::Point2f(33.5493, 92.3655));
    landmarks96.push_back(cv::Point2f(62.7299, 92.2041));

    std::string dir = "/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/112x112/";
    int pos = dir.length();
    std::vector<std::string> paths;
    paths.clear();
    dir_get_file(dir,paths);
    std::string output="/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/112x96/";
    for(int i=0;i<paths.size();i++)
    {
        std::string dirname = paths[i];
        std::cout<<dirname<<std::endl;
        cv::Mat img =cv::imread(dirname);
        if(img.empty())
            continue;
        std::string rename = dirname.replace(0,pos,output);
        int npos = rename.find_last_of('/');
        std::string pre_dir = rename.substr(0,npos);
        mkdir(pre_dir.c_str(),0755);
        cv::Mat warp_frame;
        cv::Mat warp_mat = cv::estimateRigidTransform(landmarks112, landmarks96, false);
        if (warp_mat.cols == 3 && warp_mat.rows == 2)
        {
            cv::warpAffine(img, warp_frame, warp_mat, cv::Size(96,112));
            cv::imwrite(rename,warp_frame);
        }
    }


//    std::ifstream fid("/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/fid.txt");
//    std::string outpath = "/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/112x96";
//    std::string filename;
//    while (std::getline(fid,filename)) {
//        cv::Mat img = cv::imread(filename);
//        cv::Mat warp_frame;
//        cv::Mat warp_mat = cv::estimateRigidTransform(landmarks112, landmarks96, false);
//        if (warp_mat.cols == 3 && warp_mat.rows == 2)
//        {
//            cv::warpAffine(img, warp_frame, warp_mat, cv::Size(96,112));
//            cv::imshow("img",img);
//            cv::imshow("warp_frame",warp_frame);
//            cv::waitKey(0);
//        }
//    }


}

int main()
{
    coordlandmarks.push_back(cv::Point2f(38.2946, 51.6963));
    coordlandmarks.push_back(cv::Point2f(73.5318, 51.5014));
    coordlandmarks.push_back(cv::Point2f(56.0252, 71.7366));
    coordlandmarks.push_back(cv::Point2f(41.5493, 92.3655));
    coordlandmarks.push_back(cv::Point2f(70.7299, 92.2041));

//    allign();

//    getfaceFeature();
    mtcdetec();
//    detectFace();
//    detectFrontalFace();
    //cutImgFace();
    //testVideo();
//    save_img();
    std::cout << "Hello World!" << std::endl;
    return 0;
}
