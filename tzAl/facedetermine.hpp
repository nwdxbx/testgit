#ifndef FACEDETERMINE_HPP
#define FACEDETERMINE_HPP

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include<iostream>
#include <string>

enum direction{
eDangWei = 0,
eZuoFangA = 1,
eZuoFangB = 2,
eNeiShiJing = 3,
eYiBiaoPang = 4,
eYouFangA = 5,
eYouFangB = 6,
eQianFang = 7
};

class FaceDetectermine
{
public:
    FaceDetectermine(std::string& model_file,
                     std::string& trained_file);
    ~FaceDetectermine();
    bool calcDirection(cv::Mat srcImg,cv::Rect faceRect, int &resultDir, float &frontalVal);
private:
    int predict(cv::Mat& img);
    int getMaxDir();
private:
    float _dir[8];
    caffe::shared_ptr<caffe::Net<float>> net_;
    float frontal_val;
};

FaceDetectermine::FaceDetectermine(std::string& model_file,
                                   std::string& trained_file)
{
//#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
//#else
//    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
//#endif

    /* Load the network. */
    net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_file);
}

FaceDetectermine::~FaceDetectermine()
{

}

bool FaceDetectermine::calcDirection(cv::Mat srcImg, cv::Rect faceRect, int &resultDir, float &frontalVal)
{
    cv::Mat faceImg = srcImg(faceRect).clone();
    int maxDir = predict(faceImg);
    resultDir = maxDir;
    cv::imshow("faceImg",faceImg);
    frontalVal = frontal_val;
    if(maxDir == eQianFang || maxDir == eYiBiaoPang)
        return true;
    return false;
}

int FaceDetectermine::predict(cv::Mat& img)
{
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    //step 1
    cv::Mat sample_single;
    cv::Mat m = img.clone();
    cv::resize(m, m, cv::Size(input_layer->width(), input_layer->height()));

    m = m.t();
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
    m.convertTo(sample_single, CV_32FC3);		//转浮点型
    sample_single = (sample_single - 127.5)*0.0078125;

    //step 2
    float* input_data =NULL;
    input_data = input_layer->mutable_cpu_data();

    std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;

    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += (input_layer->height())* (input_layer->width());
    }
    cv::split(sample_single, *input_channels);

    //step 3
    net_->Forward();

    //step 4
    const boost::shared_ptr<caffe::Blob<float> > feature_blob_4 = net_->blob_by_name("conv7-4");

    const float* data_ptr_4 = NULL;
    data_ptr_4 = (const float *)feature_blob_4->cpu_data();

//    const boost::shared_ptr<caffe::Blob<float> > feature_blob_3 = net_->blob_by_name("conv6-3");
//    const float* data_ptr_3 = NULL;
//    data_ptr_3 = (const float *)feature_blob_3->cpu_data();
//    float arr[10];
//    for(int i = 0; i < 10; i++)
//    {
//      if(i < 5)
//      {
//          arr[i] = data_ptr_3[i] * img.cols; //x;
//      }
//      else
//      {
//          arr[i] = data_ptr_3[i] * img.rows;      //y;
//      }
//    }
//    cv::Mat faceImg = img.clone();
//    for(int i = 0; i < 5; i++)
//    {
//        cv::Point pt;
//        pt.y = arr[i + 5];
//        pt.x = arr[i];
//        cv::circle(faceImg, pt, 2, cv::Scalar(0,255,0));
//    }
//    cv::imshow("interface",faceImg);

    const boost::shared_ptr<caffe::Blob<float> > feature_blob_5 = net_->blob_by_name("conv10_1");
    const float* data_ptr_5 = NULL;
    data_ptr_5 = (const float *)feature_blob_5->cpu_data();
    frontal_val = data_ptr_5[1] - data_ptr_5[0];
    //if(data_ptr_5[1] > data_ptr_5[0])
    std::cout<<"face deploy frontal: "<<data_ptr_5[1] <<" "<< data_ptr_5[0]<<std::endl;
    for (int i=0;i<8;i++)
    {
       _dir[i] = data_ptr_4[i];
    }
    int maxDir = getMaxDir();

    return maxDir;
}

int FaceDetectermine::getMaxDir()
{
    int maxIndex = -1;
    float maxScore = -1.0f;
    for(int i = 0; i < 8; i++)
    {
        if(_dir[i] > maxScore)
        {
            maxIndex = i;
            maxScore = _dir[i];
        }
    }
    return maxIndex;
}

#endif // FACEDETERMINE_HPP
