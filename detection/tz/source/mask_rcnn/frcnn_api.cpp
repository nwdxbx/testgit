#include "frcnn_api.hpp"

namespace FRCNN_API{

    using  namespace std;

    struct Info
    {
        float score;
        const float* head;
        int index;
    };
    bool compare(const Info& Info1, const Info& Info2)
    {
        return Info1.score > Info2.score;
    }


    void Detector::preprocess(const cv::Mat &img_in, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  CHECK(img_in.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
  DLOG(ERROR) << "img_in (CHW) : " << img_in.channels() << ", " << img_in.rows << ", " << img_in.cols; 
  input_blobs[blob_idx]->Reshape(1, img_in.channels(), img_in.rows, img_in.cols);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  const int cols = img_in.cols;
  const int rows = img_in.rows;
  for (int i = 0; i < cols * rows; i++) {
    blob_data[cols * rows * 0 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 0] ;// mean_[0]; 
    blob_data[cols * rows * 1 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 1] ;// mean_[1];
    blob_data[cols * rows * 2 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 2] ;// mean_[2];
  }
}

void Detector::preprocess(const vector<float> &data, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  input_blobs[blob_idx]->Reshape(1, data.size(), 1, 1);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  std::memcpy(blob_data, &data[0], sizeof(float) * data.size());
}

void Detector::Set_Model(std::string &proto_file, std::string &model_file) {
  this->roi_pool_layer = - 1;
//    net_.reset(new Net<float>(proto_file, caffe::TEST));
  net_.reset(new Net<float>(proto_file, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
  net_->CopyTrainedLayersFrom(model_file);
  mean_[0] = 102.9801;
  mean_[1] = 115.9465;
  mean_[2] = 122.7717;
  const vector<std::string>& layer_names = this->net_->layer_names();
  const std::string roi_name = "roi_pool";
  for (size_t i = 0; i < layer_names.size(); i++) {
    if (roi_name.size() > layer_names[i].size()) continue;
    if (roi_name == layer_names[i].substr(0, roi_name.size())) {
      CHECK_EQ(this->roi_pool_layer, -1) << "Previous roi layer : " << this->roi_pool_layer << " : " << layer_names[this->roi_pool_layer];
      this->roi_pool_layer = i;
        break;
    }
  }
  CHECK(this->roi_pool_layer >= 0 && this->roi_pool_layer < layer_names.size());
  DLOG(INFO) << "SET MODEL DONE, ROI POOLING LAYER : " << layer_names[this->roi_pool_layer];
  //caffe::Frcnn::FrcnnParam::print_param();
}

vector<boost::shared_ptr<Blob<float> > > Detector::predict(const vector<std::string> blob_names) {
  //DLOG(ERROR) << "FORWARD BEGIN";
  float loss;
  net_->Forward(&loss);
  vector<boost::shared_ptr<Blob<float> > > output;
  for (int i = 0; i < blob_names.size(); ++i) {
    output.push_back(this->net_->blob_by_name(blob_names[i]));
  }
  //DLOG(ERROR) << "FORWARD END, Loss : " << loss;
  return output;
}

void Detector::predict(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results,std::vector<cv::Mat>&masks,
                       std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps,float prob_thresh) {
  //CHECK(FrcnnParam::iter_test == -1 || FrcnnParam::iter_test > 1) << "FrcnnParam::iter_test == -1 || FrcnnParam::iter_test > 1";
//    FrcnnParam::iter_test = 5;
    thresh_ =  prob_thresh;
    if (FrcnnParam::iter_test == -1) {
    predict_original(img_in, results,masks,keypoints,detect_kps);
  } else {
    predict_iterative(img_in, results,masks,keypoints,detect_kps);
  }
}

void Detector::predict_original(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results,std::vector<cv::Mat>&masks,
                                std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps) {

//  CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, 600, 1000);
    FrcnnParam::test_nms = 0.3;

  cv::Mat img;
  const int height = img_in.rows;
  const int width = img_in.cols;
  DLOG(INFO) << "height: " << height << " width: " << width;
  img_in.convertTo(img, CV_32FC3);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      int offset = (r * img.cols + c) * 3;
      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
    }
  }
  cv::resize(img, img, cv::Size(), scale_factor, scale_factor);

  std::vector<float> im_info(3);
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  DLOG(ERROR) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
  this->preprocess(img, 0);
  this->preprocess(im_info, 1);

  vector<std::string> blob_names(4);
  blob_names[0] = "rpn_rois";
  blob_names[1] = "cls_prob";
  blob_names[2] = "bbox_pred";
  blob_names[3] = "conv_mask6";

  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
//  boost::shared_ptr<Blob<float> > rois(output[0]);
//  boost::shared_ptr<Blob<float> > cls_prob(output[1]);
//  boost::shared_ptr<Blob<float> > bbox_pred(output[2]);
//  boost::shared_ptr<Blob<float> > mask_pred(output[3]);


  //CHECK_EQ(cls_num , caffe::Frcnn::FrcnnParam::n_classes);
  results.clear();
    const float* rois = net_->blob_by_name("rpn_rois")->cpu_data();
    const float* pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
    //const int box_num = bbox_pred->num();
    const int cls_num = net_->blob_by_name("cls_prob")->channels();

//    float *boxes = NULL;
//    float *pred = NULL;
//    float *pred_per_class = NULL;
//    float *sorted_pred_cls = NULL;
//    long *keep = NULL;
//    const float* bbox_delt;
//    int num_out;
////    const float* rois;
//    float NMS_THRESH = 0.3;
////    const float* pred_cls;
//    int num;
////    rois = net_->blob_by_name("rois")->cpu_data();
////    pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
//    bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
//    num = net_->blob_by_name("rpn_rois")->num();
//    boxes = new float[num*4];
//    pred = new float[num*5*cls_num];
//    pred_per_class = new float[num*5];
//    sorted_pred_cls = new float[num*5];
//    keep = new long[num];
//
//    for (int n = 0; n < num; n++)
//    {
//        for (int c = 0; c < 4; c++)
//        {
//            boxes[n*4+c] = rois[n*5+c+1] / scale_factor;
//        }
//    }
//
//    bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, img.rows, img.cols);
//    for (int i = 1; i < cls_num; i ++)
//    {
//        for (int j = 0; j< num; j++)
//        {
//            for (int k=0; k<5; k++)
//                pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
//        }
//        boxes_sort(num, pred_per_class, sorted_pred_cls);
//        _nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, 0);
//        for(int i = 0; i < num_out; i++){
//            std::cout << keep[i] << endl;
//        }
//        //for visualize only
//        //vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH);
//    }

//  for (int cls = 1; cls < cls_num; cls++) {
//    vector<BBox<float> > bbox;
//    for (int i = 0; i < box_num; i++) {
//      float score = cls_prob->cpu_data()[i * cls_num + cls];
//
//      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
//                     rois->cpu_data()[(i * 5) + 2]/scale_factor,
//                     rois->cpu_data()[(i * 5) + 3]/scale_factor,
//                     rois->cpu_data()[(i * 5) + 4]/scale_factor);
//
//      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
//                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
//                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
//                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);
//
//      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
//      box[0] = std::max(0.0f, box[0]);
//      box[1] = std::max(0.0f, box[1]);
//      box[2] = std::min(width-1.f, box[2]);
//      box[3] = std::min(height-1.f, box[3]);
//
//      // BBox tmp(box, score, cls);
//      // LOG(ERROR) << "cls: " << tmp.id << " score: " << tmp.confidence;
//      // LOG(ERROR) << "roi: " << roi.to_string();
//      bbox.push_back(BBox<float>(box, score, cls));
//    }
//    sort(bbox.begin(), bbox.end());
//    vector<bool> select(box_num, true);
//    // Apply NMS
//     const float* data_ptr = mask_pred->cpu_data();
//      cv::Size ss(28, 28);
//    for (int i = 0; i < box_num; i++)
//      if (select[i]) {
//        if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
//        for (int j = i + 1; j < box_num; j++) {
//          if (select[j]) {
//            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
//              select[j] = false;
//            }
//          }
//        }
//        results.push_back(bbox[i]);
//        int offset = mask_pred->offset(i,cls);
//        const float* data = data_ptr+offset;
//        cv::Mat channel(ss, CV_32FC1, (float*)data);
//          masks.push_back(channel);
////        bboxs_selected.push_back(std::pair<int,int>(i,cls));
//      }
//  }

}

    /*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
    void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
    {
        vector<Info> my;
        Info tmp;
        for (int i = 0; i< num; i++)
        {
            tmp.score = pred[i*5 + 4];
            tmp.head = pred + i*5;
            my.push_back(tmp);
        }
        std::sort(my.begin(), my.end(), compare);
        for (int i=0; i<num; i++)
        {
            for (int j=0; j<5; j++)
                sorted_pred[i*5+j] = my[i].head[j];
        }
    }

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
    void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
    {
        float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
        for(int i=0; i< num; i++)
        {
            width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
            height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
            ctr_x = boxes[i*4+0] + 0.5 * width;
            ctr_y = boxes[i*4+1] + 0.5 * height;
            for (int j=0; j< class_num; j++)
            {

                dx = box_deltas[(i*class_num+j)*4+0];
                dy = box_deltas[(i*class_num+j)*4+1];
                dw = box_deltas[(i*class_num+j)*4+2];
                dh = box_deltas[(i*class_num+j)*4+3];
                pred_ctr_x = ctr_x + width*dx;
                pred_ctr_y = ctr_y + height*dy;
                pred_w = width * exp(dw);
                pred_h = height * exp(dh);
                pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, double(img_width -1)), 0.);
                pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, double(img_height -1)), 0.);
                pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, double(img_width -1)), 0.);
                pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, double(img_height -1)), 0.);
                pred[(j*num+i)*5+4] = pred_cls[i*class_num+j];
            }
        }

    }

void Detector::predict_iterative(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results,std::vector<cv::Mat>&masks,
                                 std::vector<std::vector<cv::Vec3f>>&keypoints,bool detect_kps) {

  //CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";
  CHECK(FrcnnParam::iter_test >= 1) << "iter_test should greater and queal than 1";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, 600, 1000);

//  cv::Mat img;
//  const int height = img_in.rows;
//  const int width = img_in.cols;
//  DLOG(INFO) << "height: " << height << " width: " << width;
//  img_in.convertTo(img, CV_32FC3);
//  for (int r = 0; r < img.rows; r++) {
//    for (int c = 0; c < img.cols; c++) {
//      int offset = (r * img.cols + c) * 3;
//      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
//      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
//      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
//    }
//  }
//  cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
//
//  std::vector<float> im_info(3);
//  im_info[0] = img.rows;
//  im_info[1] = img.cols;
//  im_info[2] = scale_factor;
//
//  DLOG(INFO) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
//  this->preprocess(img, 0);
//  this->preprocess(im_info, 1);
//
//  vector<std::string> blob_names(4);
//  blob_names[0] = "rpn_rois";
//  blob_names[1] = "cls_prob";
//  blob_names[2] = "bbox_pred";
//  blob_names[3] = "conv_mask6";
//
//  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
//  boost::shared_ptr<Blob<float> > rois(output[0]);
//  boost::shared_ptr<Blob<float> > cls_prob(output[1]);
//  boost::shared_ptr<Blob<float> > bbox_pred(output[2]);
//  boost::shared_ptr<Blob<float> > mask_pred(output[3]);
    cv::Mat img;
    const int height = img_in.rows;
    const int width = img_in.cols;
   // DLOG(INFO) << "height: " << height << " width: " << width;
    img_in.convertTo(img, CV_32FC3);
    img =  (img-cv::Scalar(91.4953, 103.8827, 131.0912));
//  for (int r = 0; r < img.rows; r++) {
//    for (int c = 0; c < img.cols; c++) {
//      int offset = (r * img.cols + c) * 3;
//      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
//      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
//      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
//    }
//  }
    cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
    net_->blob_by_name("data")->Reshape(1, 3, img.rows, img.cols);
    auto &input_layer=net_->input_blobs()[0];
    int w_=input_layer->width();
    int h_=input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    std::vector<cv::Mat>  input_channels;
    input_channels.resize(input_layer->shape()[0]);
    for (int j=0; j<input_layer->shape()[1]; j++)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->width()*input_layer->height();
    }

//    cv::Mat x=img.clone();
//    if(x.empty())
//        x=cv::Mat(w_,h_,CV_8UC3, cv::Scalar::all(0));
//    else
//        cv::resize(x, x, cv::Size(w_, h_), (0, 0), (0, 0), cv::INTER_CUBIC);
//    //cv::resize(x, x, cv::Size(w_, h_));
//
//    if (x.channels()==1)
//        cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);
//    cv::Mat floatImg, floatImg1;
//    x.convertTo(floatImg1, CV_32FC3);
//    preprocess(floatImg1, floatImg);
    cv::split(img, input_channels);
    float im_info[3];
    im_info[0] = img.rows;
    im_info[1] = img.cols;
    im_info[2] = scale_factor;
    net_->blob_by_name("im_info")->set_cpu_data(im_info);

    vector<std::string> blob_names;
    blob_names.push_back("rpn_rois");
    blob_names.push_back("cls_prob");
    blob_names.push_back("bbox_pred");
    blob_names.push_back("conv_mask6");
    if(detect_kps){
        blob_names.push_back("conv_kps_mask6");
    }


    vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
    boost::shared_ptr<Blob<float> > rois(output[0]);
    boost::shared_ptr<Blob<float> > cls_prob(output[1]);
    boost::shared_ptr<Blob<float> > bbox_pred(output[2]);
    boost::shared_ptr<Blob<float> > mask_pred(output[3]);
    boost::shared_ptr<Blob<float> > keypoints_detected;
    if(detect_kps){
        keypoints_detected = output[4];
    }

  const int box_num = bbox_pred->num();
  const int cls_num = cls_prob->channels();
  caffe::Frcnn::FrcnnParam::n_classes = 14;
  CHECK_EQ(cls_num , caffe::Frcnn::FrcnnParam::n_classes);
    //vector<bool> select(box_num, true);
    //vector<BBox<float> > bbox;
  std::vector<std::pair<int,int>>bboxs_selected;


  int iter_test = FrcnnParam::iter_test;
  while (--iter_test) {
    vector<Point4f<float> > new_rois;
    for (int i = 0; i < box_num; i++) { 
      int cls_mx = 1;
      for (int cls = 1; cls < cls_num; cls++) { 
        float score    = cls_prob->cpu_data()[i * cls_num + cls];
        float mx_score = cls_prob->cpu_data()[i * cls_num + cls_mx];
        if (score >= mx_score) {
          cls_mx = cls;
        }
      }

      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1],
                         rois->cpu_data()[(i * 5) + 2],
                         rois->cpu_data()[(i * 5) + 3],
                         rois->cpu_data()[(i * 5) + 4]);
#if 0
      new_rois.push_back( roi );
#endif

      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 0],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 1],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 2],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 3]);

      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
      box[0] = std::max(0.0f, box[0]);
      box[1] = std::max(0.0f, box[1]);
      box[2] = std::min(im_info[1]-1.f, box[2]);
      box[3] = std::min(im_info[0]-1.f, box[3]);

      new_rois.push_back(box);
    }
    rois->Reshape(new_rois.size(), 5, 1, 1);
    for (size_t index = 0; index < new_rois.size(); index++) {
      rois->mutable_cpu_data()[ index * 5 ] = 0;
      for (int j = 1; j < 5; j++) {
        rois->mutable_cpu_data()[ index * 5 + j ] = new_rois[index][j-1];
      }
    }

    this->net_->ForwardFrom( this->roi_pool_layer );
    if(iter_test == 2)
    {
        results.clear();
//        float* data_ptr = mask_pred->mutable_cpu_data();
//        cv::Size ss(28, 28);
        //bboxs_selected.clear();
        for (int cls = 1; cls < cls_num; cls++) {
            vector<BBox<float> > bbox;
            for (int i = 0; i < box_num; i++) {
                float score = cls_prob->cpu_data()[i * cls_num + cls];

                Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
                                   rois->cpu_data()[(i * 5) + 2]/scale_factor,
                                   rois->cpu_data()[(i * 5) + 3]/scale_factor,
                                   rois->cpu_data()[(i * 5) + 4]/scale_factor);

                Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
                                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
                                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
                                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);

                Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
                box[0] = std::max(0.0f, box[0]);
                box[1] = std::max(0.0f, box[1]);
                box[2] = std::min(width-1.f, box[2]);
                box[3] = std::min(height-1.f, box[3]);

                bbox.push_back(BBox<float>(box, i, score, cls));
            }
            sort(bbox.begin(), bbox.end());
            vector<bool> select(box_num, true);
            // Apply NMS
            for (int i = 0; i < box_num; i++)
                if (select[i]) {
//                    if (bbox[i].confidence < FrcnnParam::test_score_thresh)
                    if (bbox[i].confidence < thresh_)
                    {
                        //select[i] = false;
                        break;
                    }
                    for (int j = i + 1; j < box_num; j++) {
                        if (select[j]) {
                            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
                                select[j] = false;
                            }
                        }
                    }
                    results.push_back(bbox[i]);
//                    int offset = mask_pred->offset(i,cls-1);
//                    float* data = data_ptr+offset;
//                    cv::Mat channel(ss, CV_32FC1, data);
//                    masks.push_back(channel);
                    //std::cout << bbox[i].index << "    " << i << std::endl;
                    bboxs_selected.push_back(std::pair<int,int>(bbox[i].index,cls));
                }
        }
    }


//      if(iter_test == 2){
//          results.clear();
//          bboxs_selected.clear();
//          for (int cls = 1; cls < cls_num; cls++) {
//
//              for (int i = 0; i < box_num; i++) {
//                  float score = cls_prob->cpu_data()[i * cls_num + cls];
//
//                  Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
//                                     rois->cpu_data()[(i * 5) + 2]/scale_factor,
//                                     rois->cpu_data()[(i * 5) + 3]/scale_factor,
//                                     rois->cpu_data()[(i * 5) + 4]/scale_factor);
//
//                  Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
//                                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
//                                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
//                                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);
//
//                  Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
//                  box[0] = std::max(0.0f, box[0]);
//                  box[1] = std::max(0.0f, box[1]);
//                  box[2] = std::min(width-1.f, box[2]);
//                  box[3] = std::min(height-1.f, box[3]);
//
//                  bbox.push_back(BBox<float>(box, score, cls));
//              }
//              sort(bbox.begin(), bbox.end());
//              vector<bool> select(box_num, true);
//              // Apply NMS
//              for (int i = 0; i < box_num; i++)
//                  if (select[i]) {
//                      if (bbox[i].confidence < FrcnnParam::test_score_thresh)
//                      {
//                          //select[i] = false;
//                          break;
//                      }
//                      for (int j = i + 1; j < box_num; j++) {
//                          if (select[j]) {
//                              if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
//                                  select[j] = false;
//                              }
//                          }
//                      }
//                      results.push_back(bbox[i]);
//                      bboxs_selected.push_back(std::pair<int,int>(i,cls));
//                  }
//          }
//      }
//      if(iter_test == 1){
//          float* data_ptr = mask_pred->mutable_cpu_data();
//          cv::Size ss(28, 28);
//          for (int i = 0; i < bboxs_selected.size(); i++)
//          {
//
//              //if (select[i]) {
//                  //if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
//                  //float* data = mask_pred->mutable_cpu_data();
//                  int index = bboxs_selected[i].first;
//                  int cls = bboxs_selected[i].second;
////                  int cls = bbox[i].id;
//                  //int offset = mask_pred->offset(i,cls-1);
//                  //float* data = mask_pred->mutable_cpu_data()+offset;
//                  float* data = data_ptr+index*13*28*28+(cls-1)*28*28;
//
//                  cv::Mat channel(ss, CV_32FC1, data);
//                  masks.push_back(channel);
//              //}
//          }
//      }

    //DLOG(INFO) << "results size" << results.size() << "masks size" << masks.size();
    //DLOG(INFO) << "iter_test[" << iter_test << "] >>> rois shape : " << rois->shape_string() << "  |  cls_prob shape : " << cls_prob->shape_string() << " | bbox_pred : " << bbox_pred->shape_string();
  }
    const float* data_ptr = mask_pred->cpu_data();
    cv::Size ss(28, 28);
    for(int i=0; i < bboxs_selected.size();i++)
    {
        int cls = bboxs_selected[i].second;
        int index = bboxs_selected[i].first;
        int offset = mask_pred->offset(index,cls-1);
        //int offset = index*13*28*28+(cls-1)*28*28;
        const float* data = data_ptr+offset;
        cv::Mat mask(ss, CV_32FC1,(float*)data);
        masks.push_back(mask);
    }
    if(detect_kps){
        const float* kps_data_ptr = keypoints_detected->cpu_data();
        //cv::Size ss(28, 28);
        for(int i=0; i < bboxs_selected.size();i++)
        {
            //int cls = bboxs_selected[i].second;
            int index = bboxs_selected[i].first;
            std::vector<cv::Vec3f>key_pts;
            for(int j =0; j < 17;j++){
                int offset = keypoints_detected->offset(index,j);
                //int offset = index*13*28*28+(cls-1)*28*28;
                const float* data = kps_data_ptr+offset;
                cv::Mat keypoints_mask(ss, CV_32FC1,(float*)data);
                //masks.push_back(mask);
                double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
                cv::minMaxLoc(keypoints_mask, &minVal, &maxVal, &minLoc, &maxLoc);
                cv::Vec3f pt(maxLoc.x,maxLoc.y,maxVal);
//            for(int i =0; i < 28;i++){
//                for(int j=0; j<28;j++){
//                    std::cout << float(keypoints_mask.at<float>(i,j));
//                }
//                std::cout<<endl;
//            }
//            std::cout << "-----------------------------------------------" << endl;
                key_pts.push_back(pt);
            }
            keypoints.push_back(key_pts);
        }
    }
    //DLOG(INFO) << "results size" << results.size() << "masks size" << masks.size();
}

} // FRCNN_API
