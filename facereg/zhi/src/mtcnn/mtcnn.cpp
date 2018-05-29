// caffe
#include "mtcnn.hpp"
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <string>
#include <vector>
#include <fstream>
// opencv
#include <opencv2/opencv.hpp>


MTCnn &MTCnn::ins()
{
    static thread_local MTCnn obj;
    return obj;
}

bool MTCnn::CompareBBox(std::vector<FaceInfo>& a)
{
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; i + j < a.size() - 1; j++)
        {
            if (a[j].bbox.score < a[j + 1].bbox.score)
            {
                MTCnn::FaceInfo temp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = temp;
            }
        }
    }
    return true;
}

bool MTCnn::CompareBBox(std::vector<FaceInfo>& a, std::vector<bool>& b)
{
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; i + j < a.size() - 1; j++)
        {
            if (a[j].bbox.score < a[j + 1].bbox.score)
            {
                MTCnn::FaceInfo temp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = temp;
                bool tmpVal = b[j];
                b[j] = b[j + 1];
                b[j + 1] = tmpVal;
            }
        }
    }
    return true;
}



// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<MTCnn::FaceInfo> MTCnn::NonMaximumSuppression(std::vector<MTCnn::FaceInfo>& bboxes, float thresh, char methodType)
{
    std::vector<FaceInfo> bboxes_nms;
    int32_t select_idx = 0;
    CompareBBox(bboxes);
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged)
    {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox)
        {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        FaceRect select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++)
        {
            if (mask_merged[i] == 1)
                continue;

            FaceRect& bbox_i = bboxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

            switch (methodType)
            {
            case 'u':
                if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                    mask_merged[i] = 1;
                break;
            case 'm':
                if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                    mask_merged[i] = 1;
                break;
            default:
                break;
            }
        }
    }
    return bboxes_nms;
}

std::vector<MTCnn::FaceInfo> MTCnn::NonMaximumSuppression(std::vector<MTCnn::FaceInfo>& bboxes, float thresh,
                                                          std::vector<bool> srcFrontal, std::vector<bool>& dstFrontal)
{
    std::vector<FaceInfo> bboxes_nms;
    int32_t select_idx = 0;
    CompareBBox(bboxes, srcFrontal);
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged)
    {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox)
        {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        dstFrontal.push_back(srcFrontal[select_idx]);

        FaceRect select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++)
        {
            if (mask_merged[i] == 1)
                continue;

            FaceRect& bbox_i = bboxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;
            if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                mask_merged[i] = 1;
        }
    }
    return bboxes_nms;
}

void MTCnn::Bbox2Square(std::vector<MTCnn::FaceInfo>& bboxes)
{
    for (int i = 0; i<bboxes.size(); i++)
    {
        float h = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
        float w = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
        float side = h>w ? h : w;
        bboxes[i].bbox.x1 += (h - side)*0.5;
        bboxes[i].bbox.y1 += (w - side)*0.5;

        bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
        bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
        bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
        bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

    }
}

std::vector<MTCnn::FaceInfo> MTCnn::BoxRegress(std::vector<MTCnn::FaceInfo>& faceInfo, int stage)
{
    std::vector<FaceInfo> bboxes;
    for (int bboxId = 0; bboxId<faceInfo.size(); bboxId++)
    {
        FaceRect faceRect;
        FaceInfo tempFaceInfo;
        float regw = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
        regw += (stage == 1) ? 0 : 1;
        float regh = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
        regh += (stage == 1) ? 0 : 1;
        faceRect.y1 = faceInfo[bboxId].bbox.y1 + regw * faceInfo[bboxId].regression[0];
        faceRect.x1 = faceInfo[bboxId].bbox.x1 + regh * faceInfo[bboxId].regression[1];
        faceRect.y2 = faceInfo[bboxId].bbox.y2 + regw * faceInfo[bboxId].regression[2];
        faceRect.x2 = faceInfo[bboxId].bbox.x2 + regh * faceInfo[bboxId].regression[3];
        faceRect.score = faceInfo[bboxId].bbox.score;

        tempFaceInfo.bbox = faceRect;
        tempFaceInfo.regression = faceInfo[bboxId].regression;
        if (stage == 3)
            tempFaceInfo.facePts = faceInfo[bboxId].facePts;
        bboxes.push_back(tempFaceInfo);
    }
    return bboxes;
}



void MTCnn::Padding(int img_w, int img_h,std::vector<FaceInfo> &regressed_rects_item_i)
{
    for (int i = 0; i<regressed_rects_item_i.size(); i++)
    {
        FaceInfo tempFaceInfo;
        tempFaceInfo = regressed_rects_item_i[i];
        tempFaceInfo.bbox.y2 = (regressed_rects_item_i[i].bbox.y2 >= img_w) ? img_w-1 : regressed_rects_item_i[i].bbox.y2;
        tempFaceInfo.bbox.x2 = (regressed_rects_item_i[i].bbox.x2 >= img_h) ? img_h-1 : regressed_rects_item_i[i].bbox.x2;
        tempFaceInfo.bbox.y1 = (regressed_rects_item_i[i].bbox.y1 <1) ? 1 : regressed_rects_item_i[i].bbox.y1;
        tempFaceInfo.bbox.x1 = (regressed_rects_item_i[i].bbox.x1 <1) ? 1 : regressed_rects_item_i[i].bbox.x1;
        regressed_pading_.push_back(tempFaceInfo);
    }
}

void MTCnn::GenerateBoundingBox(caffe::Blob<float>* confidence, caffe::Blob<float>* reg, float scale,
                                  float thresh, int img_idx)
{
    int stride = 2;
    int cellSize = 12;

    int channels=confidence->shape()[1];
    int H=confidence->height();
    int W=confidence->width();

    int channels_reg=reg->shape()[1];
    int H_reg=reg->height();
    int W_reg=reg->width();

    const float* confidence_data = confidence->cpu_data();  //confidence
    const float* reg_data = reg->cpu_data();    //regession

    condidate_rects_.clear();
    for (int j=0; j<H*W; j++)
    {
        if(confidence_data[img_idx*W*H*channels + W*H*1 +j]>=thresh)
        {
            int y = j / W;
            int x = j - W * y;
            float xTop = (int)((x*stride + 1) / scale);
            float yTop = (int)((y*stride + 1) / scale);
            float xBot = (int)((x*stride + cellSize - 1 + 1) / scale);
            float yBot = (int)((y*stride + cellSize - 1 + 1) / scale);

            FaceRect faceRect;
            faceRect.x1 = xTop;
            faceRect.y1 = yTop;
            faceRect.x2 = xBot;
            faceRect.y2 = yBot;
            faceRect.score = confidence_data[img_idx*W*H*channels + W*H*1 +j];
            FaceInfo faceInfo;
            faceInfo.bbox = faceRect;
            faceInfo.regression = cv::Vec4f(reg_data[img_idx*W_reg*H_reg*channels_reg + W_reg*H_reg*0 + j],
                                            reg_data[img_idx*W_reg*H_reg*channels_reg + W_reg*H_reg*1 + j],
                                            reg_data[img_idx*W_reg*H_reg*channels_reg + W_reg*H_reg*2 + j],
                                            reg_data[img_idx*W_reg*H_reg*channels_reg + W_reg*H_reg*3 + j]);
            condidate_rects_.push_back(faceInfo);
        }
    }
}



MTCnn::MTCnn()
{
    std::string proto_model_dir="./models/mtcnn";
    threshold[0] = 0.7;
    threshold[1] = 0.8;
    //threshold[2] = 0.95;
    threshold[2] = 0.6;

    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);

    PNet_.reset(new caffe::Net<float>((proto_model_dir + "/det1.prototxt"), caffe::TEST));
    PNet_->CopyTrainedLayersFrom(proto_model_dir + "/det1.caffemodel");

    RNet_.reset(new caffe::Net<float>((proto_model_dir + "/det2.prototxt"), caffe::TEST));
    RNet_->CopyTrainedLayersFrom(proto_model_dir + "/det2.caffemodel");

    ONet_.reset(new caffe::Net<float>((proto_model_dir + "/det3.prototxt"), caffe::TEST));
    ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3.caffemodel");
}



void MTCnn::ClassifyFace_1(const std::vector<MTCnn::FaceInfo>& regressed_rects, cv::Mat &sample_single, boost::shared_ptr<caffe::Net<float> >& net, float thresh, std::vector<float> &score)
{
    int numBox = regressed_rects.size();
    caffe::Blob<float>* crop_input_layer = net->input_blobs()[0];

    int input_channel = crop_input_layer->channels();
    int input_width   = crop_input_layer->width();
    int input_height  = crop_input_layer->height();

    crop_input_layer->Reshape(numBox, input_channel, input_width, input_height);
    net->Reshape();

    condidate_rects_.clear();

    std::vector<cv::Mat> crop_imgs;
    std::vector<cv::Mat> crop_imgs_clone;
    for (int i = 0; i<numBox; i++)
    {
        int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
        int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
        int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
        int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

        cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
                                        cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));

        crop_imgs_clone.push_back(crop_img);
        cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
        crop_img.convertTo(crop_img, CV_32FC3);
        crop_img = (crop_img - cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
        crop_imgs.push_back(crop_img);
    }

    float* input_data = net->input_blobs()[0]->mutable_cpu_data();
    std::vector < std::vector<cv::Mat> > input_channels;
    input_channels.resize(numBox);
    for(int k=0;k<numBox;k++)
    {
        for (int i = 0; i < input_channel; ++i)
        {
            cv::Mat channel(input_width, input_height, CV_32FC1, input_data);
            input_channels[k].push_back(channel);
            input_data += input_width*input_height;
        }
        cv::split(crop_imgs[k], input_channels[k]);
    }

    net->Forward();
    //auto &xxx=net->output_blobs();
    const float *confidence_data = net->output_blobs()[1]->cpu_data();
    const float *reg_data = net->output_blobs()[0]->cpu_data();

    for(int i=0;i<numBox;i++)
    {
       if (confidence_data[2*i+1] > thresh)
       {
            FaceRect faceRect;
            faceRect.x1 = regressed_rects[i].bbox.x1;
            faceRect.y1 = regressed_rects[i].bbox.y1;
            faceRect.x2 = regressed_rects[i].bbox.x2;
            faceRect.y2 = regressed_rects[i].bbox.y2;
            faceRect.score = confidence_data[2*i+1];
            score.push_back(faceRect.score);
            FaceInfo faceInfo;
            faceInfo.bbox = faceRect;
            faceInfo.regression = cv::Vec4f(reg_data[i*4 + 0], reg_data[i*4 + 1], reg_data[i*4 + 2], reg_data[i*4 + 3]);
            condidate_rects_.push_back(faceInfo);
       }
    }
    regressed_pading_.clear();
}
void MTCnn::ClassifyFace_2(const std::vector<MTCnn::FaceInfo>& regressed_rects, cv::Mat &sample_single,
                         boost::shared_ptr<caffe::Net<float> >& net, float thresh, std::vector<float> &score , std::vector<bool> &frontalLabel)
{
    int numBox = regressed_rects.size();
    caffe::Blob<float>* crop_input_layer = net->input_blobs()[0];

    int input_channel = crop_input_layer->channels();
    int input_width   = crop_input_layer->width();
    int input_height  = crop_input_layer->height();

    crop_input_layer->Reshape(numBox, input_channel, input_width, input_height);
    net->Reshape();

    condidate_rects_.clear();

    std::vector<cv::Mat> crop_imgs;
    std::vector<cv::Mat> crop_imgs_clone;
    for (int i = 0; i<numBox; i++)
    {
        int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
        int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
        int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
        int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

        cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
                                        cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
        crop_imgs_clone.push_back(crop_img);
        cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
//        imshow("bs",crop_img);
//        cv::waitKey(0);
        crop_img.convertTo(crop_img, CV_32FC3);
        crop_img = (crop_img - cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
        crop_imgs.push_back(crop_img);
    }

    float* input_data = net->input_blobs()[0]->mutable_cpu_data();
    std::vector < std::vector<cv::Mat> > input_channels;
    input_channels.resize(numBox);
    for(int k=0;k<numBox;k++)
    {
        for (int i = 0; i < input_channel; ++i)
        {
            cv::Mat channel(input_width, input_height, CV_32FC1, input_data);
            input_channels[k].push_back(channel);
            input_data += input_width*input_height;
        }
        cv::split(crop_imgs[k], input_channels[k]);
    }

    net->Forward();
    //auto &xxx=net->output_blobs();
//    const float *confidence_data = net->output_blobs()[2]->cpu_data();
//    const float *reg_features = net->output_blobs()[1]->cpu_data();
//    const float *reg_box = net->output_blobs()[0]->cpu_data();
    const boost::shared_ptr<caffe::Blob<float> > feature_blob_1 = net->blob_by_name("prob1");
    const boost::shared_ptr<caffe::Blob<float> > feature_blob_2 = net->blob_by_name("conv6-2");
    const boost::shared_ptr<caffe::Blob<float> > feature_blob_3 = net->blob_by_name("conv6-3");
    const float *confidence_data = (const float *)feature_blob_1->cpu_data();
    const float *reg_features = (const float *)feature_blob_3->cpu_data();
    const float *reg_box = (const float *)feature_blob_2->cpu_data();

    const boost::shared_ptr<caffe::Blob<float> > feature_blob_4 = net->blob_by_name("prob10_1");
    const float* data_ptr_direction = NULL;
    data_ptr_direction = (const float *)feature_blob_4->cpu_data();

    const boost::shared_ptr<caffe::Blob<float> > feature_blob_5 = net->blob_by_name("conv7-4");
    const float* data_ptr_toward = NULL;
    data_ptr_toward = (const float *)feature_blob_5->cpu_data();

    for(int i=0;i<numBox;i++)
    {
       if (confidence_data[2*i+1] > thresh)
       {
            FaceRect faceRect;
            faceRect.x1 = regressed_rects[i].bbox.x1;
            faceRect.y1 = regressed_rects[i].bbox.y1;
            faceRect.x2 = regressed_rects[i].bbox.x2;
            faceRect.y2 = regressed_rects[i].bbox.y2;
            faceRect.score = confidence_data[2*i+1];
            score.push_back(faceRect.score);
            FaceInfo faceInfo;
            faceInfo.bbox = faceRect;
            faceInfo.regression = cv::Vec4f(reg_box[i*4 + 0], reg_box[i*4 + 1], reg_box[i*4 + 2], reg_box[i*4 + 3]);

            FacePts face_pts;
            float w = faceRect.y2 - faceRect.y1 + 1;
            float h = faceRect.x2 - faceRect.x1 + 1;
            for (int j = 0; j<5; j++)
            {
                face_pts.y[j] = faceRect.y1 + reg_features[i*10+j]*h-1 ;
                face_pts.x[j] = faceRect.x1 + reg_features[i*10+j+5]*w-1;
            }
            faceInfo.facePts = face_pts;

            condidate_rects_.push_back(faceInfo);

            float frontalCon = data_ptr_direction[i*2+1];

//            int max_score_dir = -1;
//            float max_score = -1.0f;

//            //std::cout << "dir score:" << std::endl;
//            for (int j=0; j<8; j++)
//            {
//                if (data_ptr_toward[i*8+j] > max_score)
//                {
//                    max_score = data_ptr_toward[i*8+j];
//                    max_score_dir = j;
//                }
//            }
            if(frontalCon > 0.5)// && max_score_dir == 7)
            {
                 frontalLabel.push_back(true);
            }
             else
            {
                frontalLabel.push_back(false);
            }
       }


    }
    regressed_pading_.clear();
}

void MTCnn::Detect(std::vector<cv::Mat> & images, std::vector<std::vector<FaceInf>> &face_Inf)
{
    int img_num = images.size();
    std::vector<int> numBox_item;
    numBox_item.resize(img_num);
    total_boxes_item.resize(img_num);
    regressed_rects_item.resize(img_num);

    std::vector<std::vector<MTCnn::FaceInfo>> face_Info;
    face_Info.resize(img_num);
    face_Inf.resize(img_num);

    std::vector<float> score1, score2;

    //pre-treated
    std::vector<cv::Mat> sample_singles;
    for(int i=0;i<img_num;i++)
    {
        cv::Mat sample_single;
        cv::cvtColor(images[i], sample_single, cv::COLOR_BGR2RGB);
        sample_single = sample_single.t();
        sample_singles.push_back(sample_single);
    }

    int height = images[0].rows;
    int width = images[0].cols;
    int minWH = std::min(height, width);
    int factor_count = 0;
    float m = 12. / minSize;
    minWH *= m;
    std::vector<float> scales;
    while (minWH >= 12)
    {
        scales.push_back(m * std::pow(factor, factor_count));
        minWH *= factor;
        ++factor_count;
    }

    for (int i = 0; i<factor_count; i++)
    {
        float scale = scales[i];
        int ws = std::ceil(height*scale);
        int hs = std::ceil(width*scale);

        std::vector<cv::Mat> resized;
        resized.resize(img_num);
        for(int ii=0;ii<img_num;ii++)
        {
            cv::resize(sample_singles[ii], resized[ii], cv::Size(ws, hs), 0, 0, cv::INTER_AREA);
            resized[ii].convertTo(resized[ii], CV_32FC3, 0.0078125, -127.5*0.0078125);
        }

        PNet_->input_blobs()[0]->Reshape(img_num, 3, hs, ws);
        PNet_->Reshape();

        float* input_data = PNet_->input_blobs()[0]->mutable_cpu_data();
        std::vector < std::vector<cv::Mat> > input_channels;
        input_channels.resize(PNet_->input_blobs()[0]->shape(0));
        for(int k=0;k<PNet_->input_blobs()[0]->shape(0);k++)
        {
            for (int i = 0; i < PNet_->input_blobs()[0]->shape(1); ++i)
            {
                cv::Mat channel(hs, ws, CV_32FC1, input_data);
                input_channels[k].push_back(channel);
                input_data += ws * hs;
            }
            cv::split(resized[k], input_channels[k]);
        }

        PNet_->Forward();

        caffe::Blob<float>* reg = PNet_->output_blobs()[0];
        caffe::Blob<float>* confidence = PNet_->output_blobs()[1];

        for(int k=0;k<img_num;k++)
        {
            GenerateBoundingBox(confidence, reg, scale, threshold[0], k);
            std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_, 0.5, 'u');
            total_boxes_item[k].insert(total_boxes_item[k].end(), bboxes_nms.begin(), bboxes_nms.end());
        }
    }

    std::vector<std::vector<bool>> isCorrectlabel;
    isCorrectlabel.resize(img_num);

    for(int id_img=0;id_img<img_num;id_img++)
    {
        numBox_item[id_img] = total_boxes_item[id_img].size();
        if (numBox_item[id_img] != 0)
        {
            total_boxes_item[id_img] = NonMaximumSuppression(total_boxes_item[id_img], 0.7, 'u');
            regressed_rects_item[id_img] = BoxRegress(total_boxes_item[id_img], 1);
            total_boxes_item[id_img].clear();
            Bbox2Square(regressed_rects_item[id_img]);
            Padding(width, height,regressed_rects_item[id_img]);

            /// Second stage
            ClassifyFace_1(regressed_rects_item[id_img], sample_singles[id_img], RNet_, threshold[1], score1);
            condidate_rects_ = NonMaximumSuppression(condidate_rects_, 0.7, 'u');
            regressed_rects_item[id_img] = BoxRegress(condidate_rects_, 2);

            Bbox2Square(regressed_rects_item[id_img]);
            Padding(width, height,regressed_rects_item[id_img]);

            /// three stage
            int thirdStageNum = 0;
            std::vector<bool> isFrontal;
            if (regressed_rects_item[id_img].size() != 0)
            {
                ClassifyFace_2(regressed_rects_item[id_img], sample_singles[id_img], ONet_, threshold[2], score2, isFrontal);
                regressed_rects_item[id_img] = BoxRegress(condidate_rects_, 3);

                //std::cout<<"face size: "<<regressed_rects_item[id_img].size()<<" "<<isFrontal.size()<<std::endl;
                std::vector<bool> frontalVal;
                face_Info[id_img] = NonMaximumSuppression(regressed_rects_item[id_img], 0.7, isFrontal, frontalVal);
                isCorrectlabel[id_img] = frontalVal;

                thirdStageNum ++;

                /*
                //showImage
                numBox_item[id_img] = face_Info[id_img].size();
                cv::Mat img3=images[id_img].clone();
                for(int i=0;i<numBox_item[id_img];i++)
                {
                    cv::Rect faceRect(face_Info[id_img][i].bbox.y1,face_Info[id_img][i].bbox.x1,
                                      face_Info[id_img][i].bbox.y2-face_Info[id_img][i].bbox.y1,
                                      face_Info[id_img][i].bbox.x2-face_Info[id_img][i].bbox.x1);
                    cv::circle(img3, cv::Point(face_Info[id_img][i].facePts.y[0],face_Info[id_img][i].facePts.x[0]), 1, cv::Scalar(0, 255, 0), 2);



                    cv::rectangle(img3, faceRect, cv::Scalar(255, 0, 0), 2);
                }
                cv::imshow("image3",img3);
                cv::waitKey(0);
                */
            }
            numBox_item[id_img] = thirdStageNum;
        }
    }


    for(int i=0;i<(int)face_Info.size();i++)
    {
        for(int j=0;j<(int)face_Info[i].size();j++)
        {
            FaceInf tmp;
            tmp.faceRect=cv::Rect(face_Info[i][j].bbox.y1,face_Info[i][j].bbox.x1,
                                      face_Info[i][j].bbox.y2-face_Info[i][j].bbox.y1,
                                      face_Info[i][j].bbox.x2-face_Info[i][j].bbox.x1);
            tmp.leye=cv::Point(face_Info[i][j].facePts.y[0],face_Info[i][j].facePts.x[0]);
            tmp.reye=cv::Point(face_Info[i][j].facePts.y[1],face_Info[i][j].facePts.x[1]);
            tmp.nose=cv::Point(face_Info[i][j].facePts.y[2],face_Info[i][j].facePts.x[2]);
            tmp.lmouth=cv::Point(face_Info[i][j].facePts.y[3],face_Info[i][j].facePts.x[3]);
            tmp.rmouth=cv::Point(face_Info[i][j].facePts.y[4],face_Info[i][j].facePts.x[4]);
            tmp.score = face_Info[i][j].bbox.score;
            tmp.isCorrect = isCorrectlabel[i][j];
            face_Inf[i].push_back(tmp);
            //cv::rectangle(images[i], face_Inf[i][j].faceRect, cv::Scalar(255, 0, 0), 2);
        }

//        cv::imshow("in_out",images[i]);
//        cv::waitKey(0);
    }

    for(int i=0;i<regressed_rects_item.size();i++)
    {
        regressed_rects_item[i].clear();
        total_boxes_item[i].clear();
    }
    regressed_pading_.clear();
    condidate_rects_.clear();
}


void MTCnn::Detect(cv::Mat &images, std::vector<FaceInf> &face_Inf)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(images);
    std::vector<std::vector<FaceInf>> res;
    Detect(pkgs, res);
    for (auto &i : res[0])
        face_Inf.push_back(i);
}

