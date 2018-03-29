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

struct det_class{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int ind;
};

std::map<int,std::string> cls_label={std::pair<int,string>(0,"background")};


int test_faster();
int test_double_faster();

int main()
{
    int sltfun = 0;
    switch (sltfun) {
    case 0:
        test_faster();
        break;
    case 1:
        test_double_faster();
        break;
    default:
        break;
    }
    cout << "Hello World!" << endl;
    return 0;
}

float get_scale_factor(int height,int width,cv::Size sz)
{
    int in_size_min = std::min(height,width);
    int in_size_max = std::max(height,width);
    float im_scale = 1.0*sz.width/in_size_min;
    if(im_scale*in_size_max>sz.height)
        im_scale = 1.0*sz.height/in_size_max;

    return im_scale;
}

void bbox_transform_inv(const cv::Scalar &rois,const cv::Scalar &deltas,cv::Scalar &result)
{
    float width = rois[2]-rois[0]+1;
    float height = rois[3]-rois[1]+1;
    float ctr_x = rois[0] + 0.5*width;
    float ctr_y = rois[1] + 0.5*height;

    float dx = deltas[0];
    float dy = deltas[1];
    float dw = deltas[2];
    float dh = deltas[3];

    float cen_x = dx*width + ctr_x;
    float cen_y = dy*height + ctr_y;
    float wid = exp(dw)*width;
    float hei = exp(dh)*height;

    result[0] = cen_x-0.5*wid;
    result[1] = cen_y-0.5*hei;
    result[2] = cen_x+0.5*wid;
    result[3] = cen_y+0.5*hei;
}

bool det_score_sort(const det_class & a, const det_class &b)
{
    return a.score > b.score;
}

float iou(det_class &det1, det_class &det2)
{
    float area1 = (det1.x2-det1.x1+1) * (det1.y2-det1.y1+1);
    float area2 = (det2.x2-det2.x1+1) * (det2.y2-det2.y1+1);
    float over_w = std::max(0.f, ((det1.x2-det1.x1+1)+(det2.x2-det2.x1+1)) - (std::max(det1.x2, det2.x2) - std::min(det1.x1, det2.x1)));
    float over_h = std::max(0.f, ((det1.y2-det1.y1+1)+(det2.y2-det2.y1+1)) - (std::max(det1.y2, det2.y2) - std::min(det1.y1, det2.y1)));

    float over_area = over_w * over_h;

    return over_area/(area1+area2);
}

void non_maxima_suppression(std::vector<det_class> &det,float fth)
{
    for (int i=0; i<det.size(); i++)
    {
        if (det[i].ind == -1)
            continue;
        for (int j=i+1; j<det.size(); j++)
        {
            if (/*det[i].ind == det[j].ind &&*/ iou(det[i], det[j]) > fth)
                det[j].ind = -1;
        }
    }
}

int test_faster()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/f/train_model/faster-rcnn/ssdtest.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/f/train_model/faster-rcnn/models/ssd_faster_rcnn_iter_40000.caffemodel");

    cv::Size sz(600,1000);
    cv::Scalar mean=cv::Scalar(127.5,127.5,127.5);

    std::ifstream flist("/media/d/big_img/image/label/dtrain.txt");
    std::string imgname;
//    std::ifstream flist("/media/d/big_img/image/label/dtrain.txt");
//    std::string lines;

    while(/*getline(flist,lines)*/flist >> imgname)
    {
//        std::stringstream strstream(lines);
//        std::string filename1,filename2;
//        strstream>>filename1>>filename2;
//        cv::Mat img = cv::imread("/media/d/big_img/image/"+filename1);
        cv::Mat img = cv::imread("/media/d/big_img/image/"+imgname);
        cv::Mat sample;
        int height = img.rows;
        int width = img.cols;
        float im_scale = get_scale_factor(height,width,sz);

        //data1
        img.convertTo(sample,CV_32FC3);
        sample -= mean;
        cv::resize(sample,sample,cv::Size(),im_scale,im_scale);
        net_->blob_by_name("data")->Reshape(1,3,sample.rows,sample.cols);
        auto &input_layer = net_->input_blobs()[0];
        float *input_data = NULL;
        input_data = input_layer->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer->channels();i++)
        {
            cv::Mat channel(input_layer->height(),input_layer->width(),CV_32FC1,input_data);
            input_channels->push_back(channel);
            input_data +=(input_layer->height())*(input_layer->width());
        }
        cv::split(sample,*input_channels);

        //im_info
        float im_info[3];
        im_info[0] = sample.rows;
        im_info[1] = sample.cols;
        im_info[2] = im_scale;
        net_->blob_by_name("im_info")->set_cpu_data(im_info);

        //forward
        //double t=(double)cv::getTickCount();
        net_->ForwardPrefilled();
       // t =((double)getTickCount()-t) / getTickFrequency();
       // std::cout<<"The color time is: "<<t<<std::endl;

        auto &cls_blob = net_->blob_by_name("cls_prob");
        auto &deltas_blob = net_->blob_by_name("bbox_pred");
        auto &rois_blob = net_->blob_by_name("rois");
        float *cls_data = (float *)cls_blob->cpu_data();
        float *deltas_data = (float *)deltas_blob->cpu_data();
        float *rois_data = (float *)rois_blob->cpu_data();
        int cls_num = cls_blob->num();
        int cls_channels = cls_blob->channels();
        int deltas_channels = deltas_blob->channels();
        int deltas_num = deltas_blob->num();
        int rois_num = rois_blob->num();
        int rois_channels = rois_blob->channels();
        std::vector<det_class> det;
        for(int i=0;i<cls_num;i++)
        {
            std::vector<std::pair<float,int> >cls_pairs;
            cls_pairs.clear();
            for(int j=0;j<cls_blob->channels();j++)
                cls_pairs.push_back(std::pair<float,int>(cls_data[i*cls_channels+j],j));
            std::sort(cls_pairs.begin(),cls_pairs.end(),[](const std::pair<float,int>& lhs,const std::pair<float,int>& rhs){
                return lhs.first>rhs.first;
            });
            int label = cls_pairs[0].second;
            float score = cls_pairs[0].first;
            if(score>0.5 && label !=0)
            {
                float rois_x1 = rois_data[i*rois_channels+1];
                float rois_y1 = rois_data[i*rois_channels+2];
                float rois_x2 = rois_data[i*rois_channels+3];
                float rois_y2 = rois_data[i*rois_channels+4];
                float deltas_dx = deltas_data[i*deltas_channels+label*4+0];
                float deltas_dy = deltas_data[i*deltas_channels+label*4+1];
                float deltas_dw = deltas_data[i*deltas_channels+label*4+2];
                float deltas_dh = deltas_data[i*deltas_channels+label*4+3];
                cv::Scalar res;
                bbox_transform_inv(cv::Scalar(rois_x1,rois_y1,rois_x2,rois_y2),cv::Scalar(deltas_dx,deltas_dy,deltas_dw,deltas_dh),res);

                det_class obj;
                obj.x1 = res[0];
                obj.y1 = res[1];
                obj.x2 = res[2];
                obj.y2 = res[3];
                obj.ind = label;
                obj.score = score;
                det.push_back(obj);
               // std::cout<<"deltas_dx: "<<deltas_dx<<" deltas_dy: "<<deltas_dy<<" deltas_dw: "<<deltas_dw<<" deltas: "<<deltas_dh<<std::endl;
            }
        }
        std::cout<<std::endl;
        std::sort(det.begin(),det.end(),det_score_sort);
        non_maxima_suppression(det, 0.2);
        for(int i=0;i<det.size();i++){
            if(det[i].ind!=-1 &&det[i].ind!=0 &&det[i].score>0.5){
                int x1=std::max(0,static_cast<int>(det[i].x1/im_scale));
                int y1=std::max(0,static_cast<int>(det[i].y1/im_scale));
                int x2=std::min(width,static_cast<int>(det[i].x2/im_scale));
                int y2=std::min(height,static_cast<int>(det[i].y2/im_scale));
                cv::rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,0,255),1);
                std::cout<<"score: "<<det[i].score<<" label: "<<det[i].ind<<std::endl;
            }
        }

        cv::imshow("img",img);
        cv::waitKey(0);
    }
}

int test_double_faster()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/f/train_model/doub_faster_rcnn/deploy.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/f/train_model/doub_faster_rcnn/vgg16_double_iter_80000.caffemodel");

    cv::Size sz(600,1000);
    cv::Scalar mean=cv::Scalar(127.5,127.5,127.5);

    std::ifstream flist("/media/f/train_model/doub_faster_rcnn/test.txt");
    std::string lines;

    while(getline(flist,lines))
    {
        std::stringstream strstream(lines);
        std::string filename1,filename2;
        strstream>>filename1>>filename2;

        cv::Mat img1 = cv::imread("/media/d/big_img/test/"+filename1);
        cv::Mat img2 = cv::imread("/media/d/big_img/test/"+filename2);
        cv::Mat sample1,sample2;
        int height = img1.rows;
        int width = img1.cols;
        float im_scale = get_scale_factor(height,width,sz);

        //data1
        img1.convertTo(sample1,CV_32FC3);
        sample1 -= mean;
        cv::resize(sample1,sample1,cv::Size(),im_scale,im_scale);
        net_->blob_by_name("data1")->Reshape(1,3,sample1.rows,sample1.cols);
        auto &input_layer1 = net_->input_blobs()[0];
        float *input_data1 = NULL;
        input_data1 = input_layer1->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels1 = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer1->channels();i++)
        {
            cv::Mat channel1(input_layer1->height(),input_layer1->width(),CV_32FC1,input_data1);
            input_channels1->push_back(channel1);
            input_data1 +=(input_layer1->height())*(input_layer1->width());
        }
        cv::split(sample1,*input_channels1);

        //data2
        img2.convertTo(sample2,CV_32FC3);
        sample2 -= mean;
        cv::resize(sample2,sample2,cv::Size(),im_scale,im_scale);
        net_->blob_by_name("data2")->Reshape(1,3,sample2.rows,sample2.cols);
        auto &input_layer2 = net_->input_blobs()[1];
        float *input_data2 = NULL;
        input_data2 = input_layer2->mutable_cpu_data();
        std::vector<cv::Mat>* input_channels2 = new std::vector<cv::Mat>;
        for(int i=0;i<input_layer2->channels();i++)
        {
            cv::Mat channel2(input_layer2->height(),input_layer2->width(),CV_32FC1,input_data2);
            input_channels2->push_back(channel2);
            input_data2 +=(input_layer2->height())*(input_layer2->width());
        }
        cv::split(sample2,*input_channels2);

        //im_info
        float im_info[3];
        im_info[0] = sample1.rows;
        im_info[1] = sample1.cols;
        im_info[2] = im_scale;
        net_->blob_by_name("im_info")->set_cpu_data(im_info);
        net_->Reshape();
        //forward
        double t=(double)cv::getTickCount();
        net_->ForwardPrefilled();
        t =((double)getTickCount()-t) / getTickFrequency();
        std::cout<<"The color time is: "<<t<<std::endl;

        auto &cls_blob1 = net_->blob_by_name("cls_prob");
        auto &deltas_blob1 = net_->blob_by_name("bbox_pred");
        auto &rois_blob1 = net_->blob_by_name("rois");
        float *cls_data1 = (float *)cls_blob1->cpu_data();
        float *deltas_data1 = (float *)deltas_blob1->cpu_data();
        float *rois_data1 = (float *)rois_blob1->cpu_data();

        int cls_num1 = cls_blob1->num();
        int cls_channels1 = cls_blob1->channels();
        int deltas_channels1 = deltas_blob1->channels();
        int deltas_num1 = deltas_blob1->num();
        int rois_num1 = rois_blob1->num();
        int rois_channels1 = rois_blob1->channels();
        //img1 process result
        std::vector<det_class> det1;
        for(int i=0;i<cls_num1;i++)
        {
            std::vector<std::pair<float,int> >cls_pairs1;
            cls_pairs1.clear();
            for(int j=0;j<cls_blob1->channels();j++)
                cls_pairs1.push_back(std::pair<float,int>(cls_data1[i*cls_channels1+j],j));
            std::sort(cls_pairs1.begin(),cls_pairs1.end(),[](const std::pair<float,int>& lhs,const std::pair<float,int>& rhs){
                return lhs.first>rhs.first;
            });
            int label = cls_pairs1[0].second;
            float score = cls_pairs1[0].first;
            if(score>0.3)
            {
                float rois_x1 = rois_data1[i*rois_channels1+1];
                float rois_y1 = rois_data1[i*rois_channels1+2];
                float rois_x2 = rois_data1[i*rois_channels1+3];
                float rois_y2 = rois_data1[i*rois_channels1+4];
                float deltas_dx = deltas_data1[i*deltas_channels1+label*4+0];
                float deltas_dy = deltas_data1[i*deltas_channels1+label*4+1];
                float deltas_dw = deltas_data1[i*deltas_channels1+label*4+2];
                float deltas_dh = deltas_data1[i*deltas_channels1+label*4+3];
                cv::Scalar res;
                bbox_transform_inv(cv::Scalar(rois_x1,rois_y1,rois_x2,rois_y2),cv::Scalar(deltas_dx,deltas_dy,deltas_dw,deltas_dh),res);

                det_class obj;
                obj.x1 = res[0];
                obj.y1 = res[1];
                obj.x2 = res[2];
                obj.y2 = res[3];
                obj.ind = label;
                obj.score = score;
                det1.push_back(obj);
            }
        }
        std::sort(det1.begin(),det1.end(),det_score_sort);
        non_maxima_suppression(det1, 0.3);
        for(int i=0;i<det1.size();i++){
            if(det1[i].ind!=-1 &&det1[i].ind!=0 &&det1[i].score>0.1){
                int x1=std::max(0,static_cast<int>(det1[i].x1/im_scale));
                int y1=std::max(0,static_cast<int>(det1[i].y1/im_scale));
                int x2=std::min(width,static_cast<int>(det1[i].x2/im_scale));
                int y2=std::min(height,static_cast<int>(det1[i].y2/im_scale));
                cv::rectangle(img1,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,0,255),1);
                std::cout<<"score1: "<<det1[i].score<<" label1: "<<det1[i].ind<<std::endl;
            }
        }

        auto &cls_blob2 = net_->blob_by_name("sec_cls_prob");
        auto &deltas_blob2 = net_->blob_by_name("sec_bbox_pred");
        auto &rois_blob2 = net_->blob_by_name("sec_rois");
        float *cls_data2 = (float *)cls_blob2->cpu_data();
        float *deltas_data2 = (float *)deltas_blob2->cpu_data();
        float *rois_data2 = (float *)rois_blob2->cpu_data();

        int cls_num2 = cls_blob2->num();
        int cls_channels2 = cls_blob2->channels();
        int deltas_channels2 = deltas_blob2->channels();
        int deltas_num2 = deltas_blob2->num();
        int rois_num2 = rois_blob2->num();
        int rois_channels2 = rois_blob2->channels();
        //img2 process result
        std::vector<det_class> det2;
        for(int i=0;i<cls_num2;i++)
        {
            std::vector<std::pair<float,int> >cls_pairs2;
            cls_pairs2.clear();
            for(int j=0;j<cls_blob2->channels();j++)
                cls_pairs2.push_back(std::pair<float,int>(cls_data2[i*cls_channels2+j],j));
            std::sort(cls_pairs2.begin(),cls_pairs2.end(),[](const std::pair<float,int>& lhs,const std::pair<float,int>& rhs){
                return lhs.first>rhs.first;
            });
            int label = cls_pairs2[0].second;
            float score = cls_pairs2[0].first;
            if(score>0.3)
            {
                float rois_x1 = rois_data2[i*rois_channels2+1];
                float rois_y1 = rois_data2[i*rois_channels2+2];
                float rois_x2 = rois_data2[i*rois_channels2+3];
                float rois_y2 = rois_data2[i*rois_channels2+4];
                float deltas_dx = deltas_data2[i*deltas_channels2+label*4+0];
                float deltas_dy = deltas_data2[i*deltas_channels2+label*4+1];
                float deltas_dw = deltas_data2[i*deltas_channels2+label*4+2];
                float deltas_dh = deltas_data2[i*deltas_channels2+label*4+3];
                cv::Scalar res;
                bbox_transform_inv(cv::Scalar(rois_x1,rois_y1,rois_x2,rois_y2),cv::Scalar(deltas_dx,deltas_dy,deltas_dw,deltas_dh),res);

                det_class obj;
                obj.x1 = res[0];
                obj.y1 = res[1];
                obj.x2 = res[2];
                obj.y2 = res[3];
                obj.ind = label;
                obj.score = score;
                det2.push_back(obj);
            }
        }
        std::sort(det2.begin(),det2.end(),det_score_sort);
        non_maxima_suppression(det2, 0.3);
        for(int i=0;i<det2.size();i++){
            if(det2[i].ind!=-1 &&det2[i].ind!=0 &&det2[i].score>0.1){
                int x1=std::max(0,static_cast<int>(det2[i].x1/im_scale));
                int y1=std::max(0,static_cast<int>(det2[i].y1/im_scale));
                int x2=std::min(width,static_cast<int>(det2[i].x2/im_scale));
                int y2=std::min(height,static_cast<int>(det2[i].y2/im_scale));
                cv::rectangle(img2,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,0,255),1);
                std::cout<<"score2: "<<det2[i].score<<" label2: "<<det2[i].ind<<std::endl;
            }
        }


        cv::imshow("img1",img1);
        cv::imshow("img2",img2);
        cv::waitKey(0);

    }
}
