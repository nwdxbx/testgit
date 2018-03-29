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

int test_SSH();

int main()
{
    int sltfun=0;
    switch (sltfun) {
    case 0:
        test_SSH();
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

//    return over_area / std::min(area1, area2);
}

void non_maxima_suppression(std::vector<det_class> &det,float fth)
{
    for (int i=0; i<det.size(); i++)
    {
        if (det[i].ind == -1)
            continue;
        for (int j=i+1; j<det.size(); j++)
        {
            if (det[i].ind == det[j].ind && iou(det[i], det[j]) > fth)
                det[j].ind = -1;
        }
    }
}


int test_SSH()
{
    Caffe::set_mode(Caffe::GPU);
    caffe::shared_ptr<caffe::Net<float> >net_;
    net_.reset(new caffe::Net<float>("/media/e/FrameWork/SSH/SSH/models/test_ssh.prototxt",TEST));
    net_->CopyTrainedLayersFrom("/media/e/FrameWork/SSH/data/SSH_models/SSH.caffemodel");

    cv::Size sz(800,1200);
    cv::Scalar mean=cv::Scalar(104,117,123);

    std::ifstream flist("/media/d/big_img/image/label/dtrain.txt");
    std::string imgname;

    while(flist >> imgname)
    {
        cv::Mat img=cv::imread("/media/d/big_img/image/"+imgname);
        cv::Mat sample;
        int height = img.rows;
        int width = img.cols;
        float im_scale = get_scale_factor(height,width,sz);

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
        float im_info[3];
        im_info[0] = sample.rows;
        im_info[1] = sample.cols;
        im_info[2] = im_scale;
        net_->blob_by_name("im_info")->set_cpu_data(im_info);

        net_->Forward();

        auto &cls_blob = net_->blob_by_name("ssh_cls_prob");
        auto &boxes_blob = net_->blob_by_name("ssh_boxes");
        int num = cls_blob->num();
        int channels= boxes_blob->channels();
        float *cls_data = (float *)cls_blob->cpu_data();
        float *boxes_data = (float *)boxes_blob->cpu_data();
        std::vector<det_class> det;
        for(int i=0;i<num;i++)
        {
            float cls = cls_data[i];
            int id = static_cast<int>(boxes_data[channels*i+0]);
            float x1 = boxes_data[channels*i+1]/im_scale;
            float y1 = boxes_data[channels*i+2]/im_scale;
            float x2 = boxes_data[channels*i+3]/im_scale;
            float y2 = boxes_data[channels*i+4]/im_scale;

            det_class obj;
            obj.score = cls;
            obj.ind = id;
            obj.x1 = x1;
            obj.y1 =y1;
            obj.x2 = x2;
            obj.y2 = y2;
            if(obj.score>0.3)
                det.push_back(obj);
        }
        std::sort(det.begin(),det.end(),det_score_sort);
        non_maxima_suppression(det, 0.3);



        for(int i=0;i<det.size();i++)
        {
            if(det[i].ind !=-1 && det[i].score>0.3)
            {
                int x1=static_cast<int>(det[i].x1);
                int y1=static_cast<int>(det[i].y1);
                int x2=static_cast<int>(det[i].x2);
                int y2=static_cast<int>(det[i].y2);
                cv::rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,0),1);
                std::cout<<det[i].score<<std::endl;
            }
        }
        cv::imshow("img",img);
        cv::waitKey(0);
    }
}
