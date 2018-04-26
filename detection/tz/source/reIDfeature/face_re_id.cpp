#include "face_re_id.hpp"
#include "../mtcnn/mtcnn.hpp"
#include "rectification.hpp"
#include "../getstructfeature/FaceLandmarkPrediction.h"
#include "../../include/tuzhenalginterface.hpp"

Face_Re_ID &Face_Re_ID::ins()
{
    static thread_local Face_Re_ID obj("./models/face_reid_p", "./models/face_reid_c");
    return obj;
}

Face_Re_ID::Face_Re_ID(const std::string &net_file,
                           const std::string &model_file)
                       : netname_(net_file),
                       modelname_(model_file)
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(gpuid);

    //net_.reset(new caffe::Net<float>(net_file, caffe::TEST));
    //net_->CopyTrainedLayersFrom(model_file);
    net_.reset(new caffe::Net<float>(net_file,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(model_file);
    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();

    MTCnn::ins();
    FaceLandmarkPrediction::ins();
}

void Face_Re_ID::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
    //resimg=(img-cv::Scalar(91.4953, 103.8827, 131.0912));
}

void Face_Re_ID::process(std::vector< cv::Mat> &imgin, std::vector<std::vector<float>> &result)
{
    //std::cout << "\t\tthe size of the network is (" << w_ << "x" << h_ << ")" << std::endl;

    auto &input_layer=net_->input_blobs()[0];
    input_layer->Reshape({(int)imgin.size(), input_layer->shape(1),
            input_layer->shape(2), input_layer->shape(3)});
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();

    std::vector<std::vector<cv::Mat>>  input_channels;
    input_channels.resize(input_layer->shape()[0]);
    for(int i=0;i<input_layer->shape()[0];i++)
    {
        for (int j=0; j<input_layer->shape()[1]; j++)
        {
            cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
            input_channels[i].push_back(channel);
            input_data += input_layer->width()*input_layer->height();
        }

        cv::Mat x=imgin[i].clone();
        if(x.empty())
            x=cv::Mat(w_,h_,CV_8UC3, cv::Scalar::all(0));
        else
            cv::resize(x, x, cv::Size(w_, h_), (0, 0), (0, 0), cv::INTER_CUBIC);
            //cv::resize(x, x, cv::Size(w_, h_));

        if (x.channels()==1)
            cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);
        cv::cvtColor(x, x, cv::COLOR_BGR2RGB);
        cv::Mat floatImg, floatImg1;
        x.convertTo(floatImg1, CV_32FC3);
        preprocess(floatImg1, floatImg);
        cv::split(floatImg, input_channels[i]);

    }
    net_->Forward();
    get_res(net_->output_blobs(), result);
}

void Face_Re_ID::clipBox(cv::Rect &rect, cv::Size imgsz)
{
    cv::Rect k;
    k.x = std::min(std::max(0, rect.x), imgsz.width-1);
    k.y = std::min(std::max(0, rect.y), imgsz.height-1);
    k.width = std::min(std::max(0, rect.x+rect.width), imgsz.width)-k.x;
    k.height = std::min(std::max(0, rect.y+rect.height), imgsz.height)-k.y;

    rect=k;
}

void Face_Re_ID::rectify(cv::Mat &img,
                        std::vector<cv::Rect> &rect,
                        std::vector<std::vector<cv::Point>> &pt,
                        std::vector<cv::Mat> &rectified_img)
{
    if (img.empty() || rect.empty() || pt.empty())
        return;
    CHECK_EQ(rect.size(), pt.size()) << "size doesn't match" << std::endl;
    rectified_img.clear();
    rectified_img.resize(rect.size());
    for (size_t i=0; i<rect.size(); i++)
    {
        cv::Rect faceR=rect[i];
        CHECK_EQ(pt[i].size(), 5) << "size doesn't match" << std::endl;
        std::vector<cv::Point2f> face_pt;
        face_pt.resize(pt[i].size());
        for (size_t j=0; j<pt[i].size(); j++)
            face_pt[j].x=pt[i][j].x, face_pt[j].y=pt[i][j].y;
        rectification<float>::ins().process(img, rectified_img[i], faceR, face_pt);
    }
}

bool Face_Re_ID::detectFace(cv::Mat& img,cv::Mat& body, cv::Point offset_pt,std::vector<cv::Mat>& faces_,void *pvTuzhenHandle,PersonInfo &personInfo)
{
    std::vector<cv::Rect> rects;
    std::vector<std::vector<cv::Point>> pts;
    std::vector<bool> iscorrect;
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle *) pvTuzhenHandle;
    int min_face_width,min_face_height;
    if (pTuzhenHandle != NULL){
        min_face_width = pTuzhenHandle->face_min_width;
        min_face_height = pTuzhenHandle->face_min_height;
    }else{
        min_face_width = 55;
        min_face_height = 65;
    }
#ifndef NDEBUG
    //save faces rect;
    boost::posix_time::ptime time_now;
    time_now=boost::posix_time::microsec_clock::universal_time();
    std::string test_file_name=to_simple_string(time_now);
    test_file_name= "./test/"+test_file_name+".jpg";
    //cv::rectangle(img,r,cv::Scalar(255,0,255),3);
    //cv::imwrite(test_file_name,img);
#endif

    float factor=1.0f;
    //cv::Mat scaleimg;
    //cv::resize(img, scaleimg, cv::Size(), factor, factor);

    std::vector<MTCnn::FaceInf> faces;
    MTCnn::ins().Detect(body, faces);
    for (size_t i=0; i<faces.size(); i++)
    {
        cv::Rect r=faces[i].faceRect;
        r.x/=factor; r.y/=factor; r.width/=factor; r.height/=factor;
        clipBox(r, img.size());

        cv::Point2f pt1, pt2, pt3, pt4, pt5;
        pt1=faces[i].leye; pt1.x/=factor; pt1.y/=factor;
        pt2=faces[i].reye; pt2.x/=factor; pt2.y/=factor;
        pt3=faces[i].nose; pt3.x/=factor; pt3.y/=factor;
        pt4=faces[i].lmouth; pt4.x/=factor; pt4.y/=factor;
        pt5=faces[i].rmouth; pt5.x/=factor; pt5.y/=factor;

//        r.x += offset_pt.x;
//        r.y += offset_pt.y;
//        pt1.x += offset_pt.x;
//        pt1.y += offset_pt.y;
//        pt2.x += offset_pt.x;
//        pt2.y += offset_pt.y;
//        pt3.x += offset_pt.x;
//        pt3.y += offset_pt.y;
//        pt4.x += offset_pt.x;
//        pt4.y += offset_pt.y;
//        pt5.x += offset_pt.x;
//        pt5.y += offset_pt.y;
        std::vector< cv::Point > pts_one{pt1, pt2, pt3, pt4, pt5};
        rects.push_back(r);
        pts.push_back(pts_one);
        iscorrect.push_back(faces[i].isCorrect);
        //debug
#ifndef NDEBUG
        cv::rectangle(img,r,cv::Scalar(255,0,255),3);
#endif
    }


    float max_score = .0f;int max_score_index = -1;
    if(rects.size() > 0){
        for(int i=0; i < rects.size();i++)
            if(max_score < faces[i].score && faces[i].isCorrect == true){
                max_score_index = i;
                max_score = faces[i].score;
            }
    }

    if(max_score_index == -1){return false;}

    //std::vector<cv::Mat> rected_face;
    std::vector<cv::Rect> face_rect;
    std::vector<std::vector<cv::Point>> face_pts;
    if(rects.size() >0 ){
        face_rect.push_back(rects[max_score_index]);
        face_pts.push_back(pts[max_score_index]);
        if (face_rect[0].height  >= min_face_height &&
                face_rect[0].width >= min_face_width){
            rectify(body, face_rect, face_pts, faces_);
        }else{
            return false;
        }

        FaceLandmarkPrediction::ins().predict(img,rects[max_score_index],personInfo);
#ifndef NDEBUG
        cv::rectangle(img,rects[max_score_index],cv::Scalar(255,0,0),3);
//        FaceLandmarkPrediction::ins().predict(img,rects[max_score_index],personInfo);
//        for(Vec2f& landmark : personInfo.face_landmarks)
//        {
//            cv::circle(img,cv::Point(int(landmark[0]),int(landmark[1])),2,cv::Scalar(255,0,0));
//        }
#endif
    }else{
        return false;
        //rected_face.push_back(body);
    }

    //std::cout << "---------------------------------------------------------";
#ifndef NDEBUG
    //cv::rectangle(img,cv::Rect(offset_pt.x,offset_pt.y,body.cols,body.rows),cv::Scalar(255,0,0),3);
    //cv::imwrite(test_file_name,img);
#endif
    return true;
}

void Face_Re_ID::process(cv::Mat &face,std::vector<float> &result)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(face);
    std::vector<std::vector<float>> res;
    process(pkgs, res);
    result=res[0];
}

float Face_Re_ID::modL2(std::vector<float>& feature)
{
    int featdim = 128;
    float clarity = 0;
    for (int i = 0; i < featdim;i++)
    {
        clarity += feature[i]*feature[i];
    }
    return clarity;
}

void Face_Re_ID::process(cv::Mat &img,cv::Mat &body,cv::Point offset_pt,std::vector<float> &result,PersonInfo &personInfo)
{
    std::vector<cv::Mat> faces;
    bool detected = detectFace(img,body,offset_pt,faces,NULL,personInfo);
    if(detected){
        std::vector< cv::Mat > pkgs;
        pkgs.push_back(faces[0]);
        std::vector<std::vector<float>> res;
        process(pkgs, res);
        result=res[0];
    }

//    if (rects.empty())
//        return;

//    size_t iwritepos=0;
//    for (size_t i=0; i<rects.size(); i++)
//    {
//        if(iscorrect.size()>0 && iscorrect[i])
//        {
//            if (iwritepos!=i)
//            {
//                rects[iwritepos]=rects[i];
//                pts[iwritepos]=pts[i];
//            }
//            ++iwritepos;
//        }
//    }
//    if (rects.size()!=iwritepos)
//    {
//        rects.resize(iwritepos);
//        pts.resize(iwritepos);
//    }
}

void Face_Re_ID::process(cv::Mat &img,cv::Mat &body,cv::Point offset_pt,std::vector<float> &result,void *pvTuzhenHandle,PersonInfo &personInfo)
{
    TuzhenalgHandle *pTuzhenHandle = (TuzhenalgHandle *) pvTuzhenHandle;
    std::vector<cv::Mat> faces;
    bool detected = detectFace(img,body,offset_pt,faces,pvTuzhenHandle,personInfo);
    if(detected){
        std::vector< cv::Mat > pkgs;
        pkgs.push_back(faces[0]);
        std::vector<std::vector<float>> res;
        process(pkgs, res);
        float clarity = modL2(res[0]);
//#ifndef NDEBUG
//        cv::rectangle(img,cv::Rect(offset_pt.x,offset_pt.y,body.cols,body.rows),cv::Scalar(255,0,0),3);
//        std::string test_file_name= "./test/"+std::to_string(clarity)+".jpg";
//        std::string test_head_file_name= "./test/"+std::to_string(clarity)+"head.jpg";
//        cv::imwrite(test_file_name,faces[0]);
//        cv::imwrite(test_head_file_name,body);
//#endif
        if(clarity > pTuzhenHandle->face_clarity_thresh){
            result=res[0];
        }
    }

//    if (rects.empty())
//        return;

//    size_t iwritepos=0;
//    for (size_t i=0; i<rects.size(); i++)
//    {
//        if(iscorrect.size()>0 && iscorrect[i])
//        {
//            if (iwritepos!=i)
//            {
//                rects[iwritepos]=rects[i];
//                pts[iwritepos]=pts[i];
//            }
//            ++iwritepos;
//        }
//    }
//    if (rects.size()!=iwritepos)
//    {
//        rects.resize(iwritepos);
//        pts.resize(iwritepos);
//    }
}

void Face_Re_ID::get_res(const std::vector<caffe::Blob<float>*> &net_res, std::vector<std::vector<float>> &res)
{
    const float *resk=net_res[0]->cpu_data();
    res.resize(net_res[0]->shape(0));
    int featdim = 128;
    for (int i=0; i<net_res[0]->shape(0); i++)
    {
        res[i].clear();
        res[i].insert(res[i].begin(), resk, resk+featdim);
        resk+=featdim;
    }

}
