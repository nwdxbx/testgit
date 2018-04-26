#include <glog/logging.h>
#include <time.h>
#include "openposeCaffe.hpp"

#include "openposeResize.hpp"
#include "openposeNms.hpp"
#include "openposeConnector.hpp"
//#include "config.hpp"


openpose_caffe &openpose_caffe::ins()
{
    static thread_local openpose_caffe obj;

    return obj;
}

openpose_caffe::openpose_caffe()
{
    std::string modelpath="./models/openpose/";
    modelsel_=property::PoseModel::COCO_18;
    netname_=modelpath+property::POSE_PROTOTXT[(int)modelsel_];
    modelname_=modelpath+property::POSE_TRAINED_MODEL[(int)modelsel_];
    outputname_="net_output";

    if (modelsel_==property::PoseModel::COCO_18)
        numofparts_=property::POSE_COCO_NUMBER_PARTS;
    else
        numofparts_=property::POSE_MPI_NUMBER_PARTS;
    

    gpu_id_=0;
    w_ = 640 / 16 * 16;
    h_ = 360 / 16 * 16;


    //zhfyuan
    //int deviceID = alg::config::ins().get_devid();
    //int deviceID = 0;

    //gpu_id_ = deviceID;

    //caffe::Caffe::SetDevice(gpu_id_);
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);

    net_.reset(new caffe::Net<float>{netname_, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"});
    net_->CopyTrainedLayersFrom(modelname_);


    net_->blobs()[0]->Reshape({1, 3, h_, w_});
    net_->Reshape();

    res_blob_ = net_->blob_by_name(outputname_);
    
    heatmap_blob_.reset(new caffe::Blob<float>(1,1,1,1));

    kernel_blob_.reset(new caffe::Blob<int>(1,1,1,1));

    peak_blob_.reset(new caffe::Blob<float>(1,1,1,1));

}


void openpose_caffe::process(const std::vector< cv::Mat > &imgs,
        std::vector< std::vector< std::vector<cv::Vec3f> > > &res)
{

    std::vector<float> scale_imgs;

    //time_t start, end;
    //start=clock();
    

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];
    input_layer->Reshape({(int)imgs.size(), input_layer->shape(1), input_layer->shape(2), input_layer->shape(3)});
    net_->Reshape();

    heatmap_blob_->Reshape({(int)imgs.size(), res_blob_->shape()[1],
        h_, w_});

    kernel_blob_->Reshape({(int)imgs.size(), res_blob_->shape()[1], 
        h_, w_});

    peak_blob_->Reshape({(int)imgs.size(), numofparts_, 
        (int)property::POSE_MAX_PEOPLE+1, 3});


    float *input_data=input_layer->mutable_cpu_data();
    std::vector< std::vector<cv::Mat> > input;
    input.resize(imgs.size());
    
    for (size_t i=0; i<imgs.size(); i++)
    {
        for (int j=0; j<input_layer->shape()[1]; j++)
        {
            cv::Mat channel(input_layer->height(), 
                    input_layer->width(), CV_32FC1, input_data);
            input[i].push_back(channel);
            input_data+=input_layer->height()*input_layer->width();
        }
        cv::Mat im=imgs[i].clone();
        float scale_im=resizeFixedAspectRatio(im, cv::Size(w_, h_), 0, cv::Scalar::all(0));
        scale_imgs.push_back(scale_im);
        if (im.channels()==1)
            cv::cvtColor(im, im, CV_GRAY2BGR);
        // cv::imshow("mx", im);
        // cv::waitKey(1);
        cv::Mat floatImg;
        im.convertTo(floatImg, CV_32FC3);
        floatImg=floatImg/256.f-0.5f;
        cv::split(floatImg, input[i]);
    }
    //end=clock();
    //std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;

    //start=clock();
    net_->Forward();
    //const float * aaaa=res_blob_->cpu_data();
    //end=clock();
    //std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;

    // for (size_t k=0; k<res_blob_->shape()[0]; k++)
    // {
    //     for (size_t i=0; i<10; i++)
    //     {
    //         for (size_t j=0; j<10; j++)
    //         {
    //             float x=*(res_blob_->cpu_data()+i*res_blob_->shape()[3]+j + 
    //                 k*res_blob_->shape()[1]*res_blob_->shape()[2]*res_blob_->shape()[3]);
    //             std::cout << x << " ";
    //         }
    //         std::cout << std::endl;

    //     }
    //     std::cout << "lasdaf"<< std::endl << std::endl << std::endl << std::endl << std::endl;
    // }

    std::array<int, 4> sz1{{heatmap_blob_->shape()[0], heatmap_blob_->shape()[1], 
        heatmap_blob_->shape()[2], heatmap_blob_->shape()[3]}};
    
    std::array<int, 4> sz2{{res_blob_->shape()[0], res_blob_->shape()[1], 
        res_blob_->shape()[2], res_blob_->shape()[3]}};

    std::array<int, 4> sz3{{peak_blob_->shape()[0], peak_blob_->shape()[1], 
        peak_blob_->shape()[2], peak_blob_->shape()[3]}};

    //start=clock();
    resizeAndMergeGpu(heatmap_blob_->mutable_gpu_data(), res_blob_->mutable_gpu_data(), 
            sz1, sz2);
    //aaaa=heatmap_blob_->cpu_data();
    //end=clock();
    //std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;

    // for (size_t k=0; k<heatmap_blob_->shape()[0]; k++)
    // {
    //     for (size_t i=100; i<110; i++)
    //     {
    //         for (size_t j=100; j<110; j++)
    //         {
    //             float x=*(heatmap_blob_->cpu_data()+i*heatmap_blob_->shape()[3]+j 
    //                 + 3*heatmap_blob_->shape()[2]*heatmap_blob_->shape()[3]+
    //                 k*heatmap_blob_->shape()[1]*heatmap_blob_->shape()[2]*heatmap_blob_->shape()[3]);
    //             std::cout << x << " ";
    //         }
    //         std::cout << std::endl;

    //     }
    //     std::cout << "lasdaf"<< std::endl << std::endl << std::endl << std::endl << std::endl;
    // }


    //const float *t=heatmap_blob_->cpu_data();

    //start=clock();
    nmsGpu(peak_blob_->mutable_gpu_data(), kernel_blob_->mutable_gpu_data(), 
            heatmap_blob_->mutable_gpu_data(), (float)property::POSE_DEFAULT_NMS_THRESHOLD[(int)modelsel_], sz3, sz1);

    //aaaa=peak_blob_->cpu_data();
    //end=clock();
    //std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;


    // for (size_t k=0; k<kernel_blob_->shape()[0]; k++)
    // {
    //     for (size_t i=110; i<120; i++)
    //     {
    //         for (size_t j=110; j<120; j++)
    //         {
    //             float x=*(kernel_blob_->cpu_data()+i*kernel_blob_->shape()[3]+j + 
    //                 + 3*kernel_blob_->shape()[2]*kernel_blob_->shape()[3]+
    //                 k*kernel_blob_->shape()[1]*kernel_blob_->shape()[2]*kernel_blob_->shape()[3]);
    //             std::cout << x << " ";
    //         }
    //         std::cout << std::endl;

    //     }
    //     std::cout << "lasdaf"<< std::endl << std::endl << std::endl << std::endl << std::endl;
    // }


    // for (size_t k=0; k<peak_blob_->shape()[0]; k++)
    // {
    //     for (size_t i=0; i<10; i++)
    //     {
    //         for (size_t j=0; j<10; j++)
    //         {
    //             float x=*(peak_blob_->cpu_data()+i*peak_blob_->shape()[3]+j + 
    //                 3*peak_blob_->shape()[2]*peak_blob_->shape()[3]+
    //                 k*peak_blob_->shape()[1]*peak_blob_->shape()[2]*peak_blob_->shape()[3]);
    //             std::cout << x << " ";
    //         }
    //         std::cout << std::endl;

    //     }
    //     std::cout << "lasdaf"<< std::endl << std::endl << std::endl << std::endl << std::endl;
    // }


    //start=clock();

    const float *heatmap=heatmap_blob_->cpu_data();
    const float *peak=peak_blob_->cpu_data();
    //const int *t2=kernel_blob_->cpu_data();
    int heatmap_offsize=heatmap_blob_->shape()[1]*
        heatmap_blob_->shape()[2]*heatmap_blob_->shape()[3];

    int peak_offsize=peak_blob_->shape()[1]*
        peak_blob_->shape()[2]*peak_blob_->shape()[3];

    res.clear();
    res.resize(imgs.size());
    for (size_t i=0; i<imgs.size(); i++)
    {
        connectBodyPartsCpu(res[i], heatmap+i*heatmap_offsize, 
                peak+i*peak_offsize, modelsel_, 
                cv::Size(heatmap_blob_->shape(3), heatmap_blob_->shape(2)), 1.f/scale_imgs[i]);
    }

    //end=clock();
    //std::cout << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
    // for (size_t i=0; i<res.size(); i++)
    // {
    //     for (size_t j=0; j<res[i].size(); j++)
    //     {
    //         for (size_t k=0; k<res[i][j].size(); k++)
    //         {
    //             cv::circle(imgs[i], cv::Point(res[i][j][k][0], res[i][j][k][1]),3,cv::Scalar(255, 0, 0), -1);\
    //         }
    //     }
    //     cv::imshow("mx", imgs[i]);
    //     cv::waitKey(1);
    // }
    //std::cout << "done" << std::endl;
}


void openpose_caffe::getmatchpair(std::vector<std::pair<size_t, size_t> > &matchpair)
{
    if (modelsel_==property::PoseModel::COCO_18)
    {
        auto v=property::POSE_COCO_PAIRS_RENDER;
        for (size_t i=0; i<v.size()/2; i++)
            matchpair.push_back({v[2*i], v[2*i+1]});
    }
    else
    {
        auto v=property::POSE_MPI_PAIRS_RENDER;
        for (size_t i=0; i<v.size()/2; i++)
            matchpair.push_back({v[2*i], v[2*i+1]});
    }
}



float openpose_caffe::resizeFixedAspectRatio(cv::Mat& cvMat,
        const cv::Size& targetSize, const int borderMode, const cv::Scalar& borderValue)
{

    cv::Size initialSize=cvMat.size();
    cv::Mat M = cv::Mat::eye(2,3,CV_32FC1);
    float ratioWidth,ratioHeight,scaleFactor;
    ratioWidth = targetSize.width / (float)initialSize.width;
    ratioHeight = targetSize.height / (float)initialSize.height;
    scaleFactor=std::min(ratioWidth, ratioHeight);
    M.at<float>(0,0) = scaleFactor;
    M.at<float>(1,1) = scaleFactor;
    if (scaleFactor != 1. || targetSize != cvMat.size())
        cv::warpAffine(cvMat, cvMat, M, targetSize, (scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC), borderMode, borderValue);
    return scaleFactor;
}


void openpose_caffe::process(cv::Mat &img, 
        std::vector< std::vector< cv::Vec3f> > &res)
{
    std::vector<cv::Mat> imgs;
    std::vector<std::vector<std::vector<cv::Vec3f> > > ress;
    imgs.push_back(img);
    process(imgs, ress);
    res=ress[0];
}
