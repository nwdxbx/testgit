#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;

int test_Picture();
int test_list();
int test_segnet();
int test_deeplabV2();
int test_ssd_clip_border();

static int colormap[] =
{
    0,0,0,
    0,0,255,
    0,255,0,
    255,0,0,
    0,255,255,
    255,0,255,
    255,255,0,
    125,255,0,
    125,0,255,
    0,125,255,
    0,0,125,
    0,125,0,
    125,0,0,
    0,125,125,
    125,0,125,
    125,125,0,
    255,125,0,
//    255,0,125,
//    0,255,125,
    255,255,255
};


int main()
{
    int sltfun=3;
    switch (sltfun) {
    case 0:
        test_Picture();
        break;
    case 1:
        test_list();
        break;
    case 2:
        test_segnet();
        break;
    case 3:
        test_ssd_clip_border();
        break;
    default:
        break;
    }

    cout << "Hello World!" << endl;
    return 0;
}

int test_ssd_clip_border()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/f/train_model/realface/renet3/FPNdeploy.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/f/train_model/realface/renet3/models/fpn_iter_50000.caffemodel");

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);
    float scale =1.0;
//    cv::Scalar mean=cv::Scalar(127.5,127.5,127.5);
//    float scale = 0.0078125;
    std::ifstream flist("/media/f/src_data/Face/Test/list1.txt");
    std::string imgname;
    while(flist >> imgname)
    {
        cv::Mat img=cv::imread(imgname);

        cv::Mat dstimage;
        int height = img.rows;
        int width = img.cols;
        int abs_diff = std::abs(2*width-height);
        if(2*width<height)
        {
            cv::Rect rect(0,0,width,2*width);
            dstimage = img(rect).clone();
        }
        else
            cv::copyMakeBorder(img,dstimage,0,abs_diff,0,0,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));

        cv::Mat sample,sample_single;
        if (dstimage.channels() == 3 && num_channels == 1)
            cv::cvtColor(dstimage, sample, cv::COLOR_BGR2GRAY);
        else if (dstimage.channels() == 4 && num_channels == 1)
            cv::cvtColor(dstimage, sample, cv::COLOR_BGRA2GRAY);
        else if (dstimage.channels() == 4 && num_channels == 3)
            cv::cvtColor(dstimage, sample, cv::COLOR_BGRA2BGR);
        else if (dstimage.channels() == 1 && num_channels == 3)
            cv::cvtColor(dstimage, sample, cv::COLOR_GRAY2BGR);
        else
            sample = dstimage.clone();
        cv::resize(sample,sample,cv::Size(input_layer->width(),input_layer->height()));
        sample.convertTo(sample_single,CV_32FC3);
        sample_single = (sample_single-mean)*scale;

        float *input_data=NULL;
        input_data=input_layer->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer->channels();i++)
        {
            cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
            input_channels->push_back(channel);
            input_data +=(input_layer->height())*(input_layer->width());
        }
        cv::split(sample_single,*input_channels);

        double start = cv::getTickCount();
        net_->Forward();
        double end = cv::getTickCount();
//        std::cout<<"The Time: "<<1000.0*(end-start)/cv::getTickFrequency()<<" ms"<<std::endl;
        Blob<float>* result_blob = net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        const int num_det = result_blob->height();
        vector<vector<float> > detections;
        for (int k = 0; k < num_det; ++k) {
            if (result[0] == -1) {
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
        for(int i=0;i<detections.size();i++)
        {
            const vector<float>& d = detections[i];
            const float score = d[2];
//            std::string strscore;
//            stringstream instr;
//            instr<<score;
//            instr>>strscore;
            if(score >= 0.6)
            {
                int xmin = static_cast<int>(d[3] * dstimage.cols);
                int ymin = static_cast<int>(d[4] * dstimage.rows);
                int xmax = static_cast<int>(d[5] * dstimage.cols);
                int ymax = static_cast<int>(d[6] * dstimage.rows);
                int label = static_cast<int>(d[1]);
//                std::string str_label;
//                std::stringstream ofs;
//                ofs<<label;
//                ofs>>str_label;
                cv::rectangle(dstimage,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin),cv::Scalar(0,255,0),2,8);
                std::cout<<"score: "<<score<<" label: "<<label<<std::endl;
//                cv::putText(img,str_label,cv::Point(xmin+10,ymin+10),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(255,0,0),1);
//                std::cout<<"rect width: "<<xmax-xmin<<" rect height: "<<ymax-ymin<<std::endl;
            }
        }
//        std::cout<<"width: "<<dstimage.cols<<" height: "<<dstimage.rows<<" ratio: "<<1.0*dstimage.rows/dstimage.cols<<std::endl;
        cv::imshow("img",dstimage);
        char c=cv::waitKey(0);
        if(c == 27) break;
    }

    return 0;
}

int test_segnet()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/f/train_model/segnet/segnet_deploy",TEST));
    net_->CopyTrainedLayersFrom("/media/f/train_model/segnet/art_model/segnet_iter_40000.caffemodel");

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);

//    std::ifstream flist("/media/d/list.txt");
    std::ifstream flist("/media/f/src_data/Face/1214/list.txt");
    std::string imgname;
    while(flist >> imgname)
    {
        cv::Mat img=cv::imread(imgname);
        cv::Mat sample,sample_single;
        if (img.channels() == 3 && num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4 && num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        else if (img.channels() == 1 && num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        else
            sample = img.clone();
        cv::resize(sample,sample,cv::Size(input_layer->width(),input_layer->height()));
        cv::Mat showimg = sample.clone();
        sample.convertTo(sample_single,CV_32FC3);
//        sample_single -=mean;

        float *input_data=NULL;
        input_data=input_layer->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer->channels();i++)
        {
            cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
            input_channels->push_back(channel);
            input_data +=(input_layer->height())*(input_layer->width());
        }
        cv::split(sample_single,*input_channels);

        double start = cv::getTickCount();
        net_->Forward();
        Blob<float>* output_layer = net_->output_blobs()[0];
        float *data_ptr = (float *)output_layer->cpu_data();
        cv::Mat colorimg = showimg.clone();
        for(size_t i=0;i<showimg.rows;i++)
            for(size_t j=0;j<showimg.cols;j++)
            {
                int id =-1;
                float max = -1.0;
                for(int n=0;n <18;n++)
                {
                    if (data_ptr[n*showimg.cols*showimg.rows+i*showimg.cols+j] > max)
                    {
                        max = data_ptr[n*showimg.cols*showimg.rows+i*showimg.cols+j];
                        id = n;
                    }
                }

                colorimg.at<cv::Vec3b>(i,j)[0] = colormap[id*3];
                colorimg.at<cv::Vec3b>(i,j)[1] = colormap[id*3+1];
                colorimg.at<cv::Vec3b>(i,j)[2] = colormap[id*3+2];
            }
        cv::imshow("showimg",showimg);
        cv::imshow("color",colorimg);
        char c=cv::waitKey(0);
        if(c == 27) break;
    }

    return 0;
}

int test_list()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/f/train_model/realface/renetVOC/deploy.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/f/train_model/realface/renetVOC/models/VGGNet/VOC_iter_120000.caffemodel");

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);

    std::ifstream flist("/media/e/FrameWork/caffe-ssd/examples/images/list.txt");
//    std::ifstream flist("/media/f/src_data/Face/Test/list.txt");
    std::string imgname;
    while(flist >> imgname)
    {
        cv::Mat img=cv::imread(imgname);
        cv::Mat sample,sample_single;
        if (img.channels() == 3 && num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4 && num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        else if (img.channels() == 1 && num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        else
            sample = img.clone();
        cv::resize(sample,sample,cv::Size(input_layer->width(),input_layer->height()));
        sample.convertTo(sample_single,CV_32FC3);
        sample_single -=mean;

        float *input_data=NULL;
        input_data=input_layer->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer->channels();i++)
        {
            cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
            input_channels->push_back(channel);
            input_data +=(input_layer->height())*(input_layer->width());
        }
        cv::split(sample_single,*input_channels);

        double start = cv::getTickCount();
        net_->Forward();
        double end = cv::getTickCount();
        std::cout<<"The Time: "<<1000.0*(end-start)/cv::getTickFrequency()<<" ms"<<std::endl;
        Blob<float>* result_blob = net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        const int num_det = result_blob->height();
        vector<vector<float> > detections;
        for (int k = 0; k < num_det; ++k) {
            if (result[0] == -1) {
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
        for(int i=0;i<detections.size();i++)
        {
            const vector<float>& d = detections[i];
            const float score = d[2];
            std::string strscore;
            stringstream instr;
            instr<<score;
            instr>>strscore;
            if(score >= 0.4)
            {
                int xmin = static_cast<int>(d[3] * img.cols);
                int ymin = static_cast<int>(d[4] * img.rows);
                int xmax = static_cast<int>(d[5] * img.cols);
                int ymax = static_cast<int>(d[6] * img.rows);
                int label = static_cast<int>(d[1]);
                std::string str_label;
                std::stringstream ofs;
                ofs<<label;
                ofs>>str_label;
                cv::rectangle(img,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin),cv::Scalar(0,0,255),2,8);
//                cv::putText(img,str_label,cv::Point(xmin+10,ymin+10),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(255,0,0),1);
//                std::cout<<"rect width: "<<xmax-xmin<<" rect height: "<<ymax-ymin<<std::endl;
            }
        }
        std::cout<<"width: "<<img.cols<<" height: "<<img.rows<<" ratio: "<<1.0*img.rows/img.cols<<std::endl;
        cv::imshow("img",img);
        char c=cv::waitKey(0);
        if(c == 27) break;
    }

    return 0;
}

int test_Picture()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/xbx/95f9261f-a694-4aad-abfd-ac1f358df649/train_model/fddb/deploy.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/xbx/95f9261f-a694-4aad-abfd-ac1f358df649/train_model/fddb/models/VGGNet/VGG_VOC0712_SSD_300x300_iter_10000.caffemodel");

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    int num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    cv::Scalar mean=cv::Scalar(104,117,123);
    cv::Mat img=cv::imread("/media/xbx/95f9261f-a694-4aad-abfd-ac1f358df649/src_data/Face/fdbb/JPEGImages/fddb_28.jpg");
    cv::Mat sample,sample_single;
    if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img.clone();
    cv::resize(sample,sample,cv::Size(input_layer->width(),input_layer->height()));
    sample.convertTo(sample_single,CV_32FC3);
    sample_single -=mean;

    float *input_data=NULL;
    input_data=input_layer->mutable_cpu_data();
    std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
    for(int i=0;i<input_layer->channels();i++)
    {
        cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
        input_channels->push_back(channel);
        input_data +=(input_layer->height())*(input_layer->width());
    }
    cv::split(sample_single,*input_channels);

    net_->Forward();
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    for(int i=0;i<detections.size();i++)
    {
        const vector<float>& d = detections[i];
        const float score = d[2];
        std::string strscore;
        stringstream instr;
        instr<<score;
        instr>>strscore;
        if(score >= 0.6)
        {
            int xmin = static_cast<int>(d[3] * img.cols);
            int ymin = static_cast<int>(d[4] * img.rows);
            int xmax = static_cast<int>(d[5] * img.cols);
            int ymax = static_cast<int>(d[6] * img.rows);
            cv::rectangle(img,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin),cv::Scalar(0,0,255),2,8);
            cv::putText(img,strscore,cv::Point(xmin,ymin),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(255,0,0),1);
        }
    }
    cv::imshow("img",img);
    cv::waitKey(0);

    return 0;
}
