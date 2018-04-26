//
// Created by zht on 17-10-30.
//

#include "face_interface.h"
#include "FaceLandmarkPrediction.h"
#include "ssd.hpp"


face_interface &face_interface::ins() {
    static thread_local face_interface obj;
    return obj;
}

void face_interface::setGPUID(int gpu_id) {
    gpu_id_ = gpu_id;
}

face_interface::face_interface() {
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);

    // init face_caffe_model
//    const string net_file = "./models/face_attrs_deploy_p";  // face_attrs_deploy.prototxt
//    net_.reset(new caffe::Net<float>(net_file, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));  //face_attrs.caffemodel
//    net_->CopyTrainedLayersFrom(
//            "./models/face_attrs_c");

//    const string net_file = "./models/face_attrs_deploy.prototxt";  // face_attrs_deploy.prototxt
//    net_.reset(new caffe::Net<float>(net_file, caffe::TEST));  //face_attrs.caffemodel
//    net_->CopyTrainedLayersFrom(
//            "./models/face_attrs.caffemodel");

    const string net_file = "./models/face_attrs_p";
    net_.reset(new caffe::Net<float>(net_file, caffe::TEST, "B743382C96DB85858FF65AE97DF98970", "802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(
            "./models/face_attrs_c");

    const string age_net_file = "./models/age_p";
    age_net_.reset(
            new caffe::Net<float>(age_net_file, caffe::TEST, "B743382C96DB85858FF65AE97DF98970", "802030C17462D3E3"));
    age_net_->CopyTrainedLayersFrom(
            "./models/age_c");

    const string gender_net_file = "./models/gender_p";
    gender_net_.reset(new caffe::Net<float>(gender_net_file, caffe::TEST, "B743382C96DB85858FF65AE97DF98970",
                                            "802030C17462D3E3"));
    gender_net_->CopyTrainedLayersFrom(
            "./models/gender_c");

    ssd_detect_caffe::ins();
    //FaceLandmarkPrediction::ins();
}

void face_interface::get_head_infos(cv::Mat &input_img, PersonInfo &personInfo, float attr_thres,
                                    float head_thres) {


    // 把size过滤加上 40*40
    //ssd_detect_caffe::ins();

    ssd_detres detect_box = {-1, -1, -1, -1, -1, -1};
    cv::Mat input = input_img.clone();
    ssd_detect_caffe::ins().process(input, detect_box, head_thres);

    // init --> face unknown
    personInfo.head_type = EM_PERSON_HEAD_UNKNOWN;
    personInfo.bgetface = false;

    if (input_img.cols <= 40 || input_img.rows <= 40)
        return;

    if (detect_box.label == 1) {
        personInfo.head_type = EM_PERSON_HEAD_FRONT;
    } else if (detect_box.label == 2) {
        personInfo.head_type = EM_PERSON_HEAD_BACK;
    } else {
        personInfo.head_type = EM_PERSON_HEAD_UNKNOWN;
    }

    if (detect_box.score != -1 && personInfo.head_type == EM_PERSON_HEAD_FRONT) {
        if (detect_box.x1 < 0) detect_box.x1 = 0;
        if (detect_box.y1 < 0) detect_box.y1 = 0;

        int width =
                detect_box.x2 - input_img.cols < 0 ? (detect_box.x2 - detect_box.x1) : (input_img.cols - detect_box.x1);
        int height =
                detect_box.y2 - input_img.rows < 0 ? (detect_box.y2 - detect_box.y1) : (input_img.rows - detect_box.y1);

        // 检测到头部
        cv::Rect rect(detect_box.x1, detect_box.y1, width, height);

        // front face
        personInfo.bgetface = true;
        personInfo.head_type = EM_PERSON_HEAD_FRONT;
        personInfo.facerct = rect;

        cv::Mat body_img = input_img.clone();
        cv::Mat head_img = input_img.clone()(rect);
        predict_facefields(head_img, body_img, personInfo, attr_thres);

    } else if (detect_box.score != -1 && personInfo.head_type == EM_PERSON_HEAD_BACK) {
        // 背头
        if (detect_box.x1 < 0) detect_box.x1 = 0;
        if (detect_box.y1 < 0) detect_box.y1 = 0;

        int width =
                detect_box.x2 - input_img.cols < 0 ? (detect_box.x2 - detect_box.x1) : (input_img.cols - detect_box.x1);
        int height =
                detect_box.y2 - input_img.rows < 0 ? (detect_box.y2 - detect_box.y1) : (input_img.rows - detect_box.y1);

        // 检测到头部
        cv::Rect rect(detect_box.x1, detect_box.y1, width, height);

        // front face
        personInfo.bgetface = true;
        personInfo.head_type = EM_PERSON_HEAD_BACK;
        personInfo.facerct = rect;
        //FaceLandmarkPrediction::ins().predict(head_img,personInfo);
    }
}

void face_interface::predict_facefields(vector<cv::Mat> &input_head_imgs, std::vector<cv::Mat> &input_body_imgs,
                                        vector<PersonInfo> &personInfos,
                                        caffe::shared_ptr<caffe::Net<float>> cur_net_, std::string net_tag,
                                        float attr_thres, EGender_Info gender_info) {

    PersonInfo personInfo;

    if (net_tag != "gender") {

        auto input_layer_ = cur_net_->input_blobs()[0];
        w_ = input_layer_->width();
        h_ = input_layer_->height();

        input_layer_->Reshape({(int) input_head_imgs.size(), input_layer_->shape(1),
                               input_layer_->shape(2), input_layer_->shape(3)});
        cur_net_->Reshape();

        float *input_data = input_layer_->mutable_cpu_data();

        std::vector<std::vector<cv::Mat>> input_channels;
        input_channels.resize(input_layer_->shape()[0]);

        for (int i = 0; i < input_layer_->shape()[0]; ++i) {
            for (int j = 0; j < input_layer_->shape()[1]; ++j) {
                cv::Mat channel(input_layer_->height(), input_layer_->width(), CV_32FC1, input_data);
                input_channels[i].push_back(channel);

                input_data += input_layer_->width() * input_layer_->height();
            }

            cv::Mat x = input_head_imgs[i].clone();
            if (x.empty()) {
                x = cv::Mat(w_, h_, CV_8UC3, cv::Scalar::all(0));
            } else {
                cv::resize(x, x, cv::Size(w_, h_));
            }

            if (x.channels() == 1) cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);

            cv::Mat floatImg, floatImg1;
            x.convertTo(floatImg1, CV_32FC3);

            floatImg = floatImg1.clone();

            cv::Scalar mean(127.5, 127.5, 127.5);
            floatImg -= mean; // minus mean value
            floatImg /= 128;

            cv::split(floatImg, input_channels[i]);
        }


        cur_net_->Forward();

    } else {
        // gender

        auto input_layer_ = cur_net_->input_blobs()[0];
        w_ = input_layer_->width();
        h_ = input_layer_->height();

        input_layer_->Reshape({(int) input_head_imgs.size(), input_layer_->shape(1),
                               input_layer_->shape(2), input_layer_->shape(3)});

        float *input_data = input_layer_->mutable_cpu_data();

        std::vector<std::vector<cv::Mat>> input_channels;
        input_channels.resize(input_layer_->shape()[0]);

        for (int i = 0; i < input_layer_->shape()[0]; ++i) {
            for (int j = 0; j < input_layer_->shape()[1]; ++j) {
                cv::Mat channel(input_layer_->height(), input_layer_->width(), CV_32FC1, input_data);
                input_channels[i].push_back(channel);

                input_data += input_layer_->width() * input_layer_->height();
            }

            cv::Mat x = input_head_imgs[i].clone();
            if (x.empty()) {
                x = cv::Mat(w_, h_, CV_8UC3, cv::Scalar::all(0));
            } else {
                cv::resize(x, x, cv::Size(w_, h_));
            }

            if (x.channels() == 1) cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);

            cv::Mat floatImg, floatImg1;
            x.convertTo(floatImg1, CV_32FC3);

            floatImg = floatImg1.clone();

            cv::split(floatImg, input_channels[i]);
        }

        auto input_layer_body_ = cur_net_->input_blobs()[1];
        w_ = input_layer_body_->width();
        h_ = input_layer_body_->height();

        input_layer_body_->Reshape({(int) input_body_imgs.size(), input_layer_body_->shape(1),
                                    input_layer_body_->shape(2), input_layer_body_->shape(3)});

        float *input_data_body = input_layer_body_->mutable_cpu_data();

        std::vector<std::vector<cv::Mat>> input_channels_body;
        input_channels_body.resize(input_layer_body_->shape()[0]);

        for (int i = 0; i < input_layer_body_->shape()[0]; ++i) {
            for (int j = 0; j < input_layer_body_->shape()[1]; ++j) {
                cv::Mat channel(input_layer_body_->height(), input_layer_body_->width(), CV_32FC1, input_data_body);
                input_channels_body[i].push_back(channel);

                input_data_body += input_layer_body_->width() * input_layer_body_->height();
            }

            cv::Mat x = input_body_imgs[i].clone();
            if (x.empty()) {
                x = cv::Mat(w_, h_, CV_8UC3, cv::Scalar::all(0));
            } else {
                cv::resize(x, x, cv::Size(w_, h_));
            }

            if (x.channels() == 1) cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);

            cv::Mat floatImg, floatImg1;
            x.convertTo(floatImg1, CV_32FC3);

            floatImg = floatImg1.clone();

            cv::split(floatImg, input_channels_body[i]);
        }

        cur_net_->Forward();
    }


    if (net_tag == "gender") {
        // 男女性别
        auto male = cur_net_->blob_by_name("gender_prob");
        auto male_ptr = male->cpu_data();

        vector<float> male_vec = {male_ptr[0], male_ptr[1]};
        auto male_value = std::max_element(male_vec.begin(), male_vec.end());
        int male_index = std::distance(male_vec.begin(), male_value);

        if (*male_value > attr_thres) {
            if (male_index == 1)
                personInfo.gender_info = EM_GENDER_MALE;
            else if (male_index == 0)
                personInfo.gender_info = EM_GENDER_FEMALE;

            personInfo.gender_score = *male_value;
        } else {
            personInfo.gender_info = EM_GENDER_UNKNOWN;
        }

    } else if (net_tag == "age") {
        // 年龄
        auto age_type = cur_net_->blob_by_name("age/softmax");
        auto age_type_ptr = age_type->cpu_data();

        vector<float> age_vec = {age_type_ptr[0], age_type_ptr[1], age_type_ptr[2]};
        auto age_value = std::max_element(age_vec.begin(), age_vec.end());
        int age_index = std::distance(age_vec.begin(), age_value);

        if (*age_value > attr_thres) {
            if (age_index == 0) {
                personInfo.age_info = EM_AGE_YOUTH; // 青少年 0～18
                personInfo.age_value = -1;

            } else if (age_index == 2) {
                personInfo.age_info = EM_AGE_OLD; // 老年 55～
                personInfo.age_value = -1;

            } else if (age_index == 1) {
                personInfo.age_info = EM_AGE_MIDDLE_AGE; // 中青年18～55

                // 年龄具体值
                auto age = cur_net_->blob_by_name("age/regression"); // age/regression
                auto age_ptr = (int) age->cpu_data()[0];
                personInfo.age_value = age_ptr;
            }
        }
    } else if (net_tag == "common") {

        // 是否帽子
        auto hat = cur_net_->blob_by_name("loss_hat/loss_hat");
        auto hat_ptr = hat->cpu_data();

        vector<float> hat_vec = {hat_ptr[0], hat_ptr[1]};
        auto hat_value = std::max_element(hat_vec.begin(), hat_vec.end());
        int hat_index = std::distance(hat_vec.begin(), hat_value);

        if (*hat_value > attr_thres) {
            if (hat_index == 1) {
                if (*hat_value > 0.75) {
                    personInfo.hat_info = EM_HAT_TRUE;
                } else {
                    personInfo.hat_info = EM_HAT_FALSE;
                }

            } else if (hat_index == 0) {
                personInfo.hat_info = EM_HAT_FALSE;
            }

            personInfo.hat_score = *hat_value;
        } else {
            personInfo.hat_info = EM_HAT_UNKNOWN;
        }

        // 是否刘海
        auto bang = net_->blob_by_name("loss_bang/loss_bang");
        auto bang_ptr = bang->cpu_data();

        vector<float> bang_vec = {bang_ptr[0], bang_ptr[1]};
        auto bang_value = std::max_element(bang_vec.begin(), bang_vec.end());
        int bang_index = std::distance(bang_vec.begin(), bang_value);

        if (*bang_value > attr_thres) {

            if (bang_index == 1) {
                if (*bang_value > 0.8) {
                    personInfo.fringe_info = EM_FRINGE_TRUE;
                } else {
                    personInfo.fringe_info = EM_FRINGE_FALSE;
                }

            } else if (bang_index == 0) {
                personInfo.fringe_info = EM_FRINGE_FALSE;
            }

            personInfo.fringe_score = *bang_value;
        } else {
            personInfo.fringe_info = EM_FRINGE_UNKNOWN;
        }

        // 是否光头
        auto bald = net_->blob_by_name("loss_bald/loss_bald");
        auto bald_ptr = bald->cpu_data();

        vector<float> bald_vec = {bald_ptr[0], bald_ptr[1]};
        auto bald_value = std::max_element(bald_vec.begin(), bald_vec.end());
        int bald_index = std::distance(bald_vec.begin(), bald_value);

        if (*bald_value > attr_thres) {

            //如果是女性，则阈值设高0.9
            if (gender_info == EM_GENDER_FEMALE) {
                if (bald_index == 1 && *bald_value > 0.9) {
                    personInfo.bare_info = EM_BARE_TRUE;
                } else {
                    personInfo.bare_info = EM_BARE_FALSE;
                }

            } else {
                // 男性, 因光头数据过少，所以设高阈值，提高召回率
                if (bald_index == 1 && *bald_value > 0.8) {
                    personInfo.bare_info = EM_BARE_TRUE;
                } else {
                    personInfo.bare_info = EM_BARE_FALSE;
                }
            }

            personInfo.bare_score = *bald_value;
        } else {
            personInfo.bare_info = EM_BARE_UNKNOWN;
        }

        // 是否口罩
        auto mask = net_->blob_by_name("loss_mask/loss_mask");
        auto mask_ptr = mask->cpu_data();

        vector<float> mask_vec = {mask_ptr[0], mask_ptr[1]};
        auto mask_value = std::max_element(mask_vec.begin(), mask_vec.end());
        int mask_index = std::distance(mask_vec.begin(), mask_value);

        if (*mask_value > attr_thres) {
            // 把口罩阈值设高
            if (mask_index == 1) {
                if (*mask_value > 0.8) {
                    personInfo.mask_info = EM_MASK_TRUE;
                } else {
                    personInfo.mask_info = EM_MASK_FALSE;
                }

            } else if (mask_index == 0) {
                personInfo.mask_info = EM_MASK_FALSE;
            }

        } else {
            personInfo.mask_info = EM_MASK_UNKNOWN;
        }

        // 是否眼镜
        auto glass = net_->blob_by_name("loss_glass/loss_glass");
        auto glass_ptr = glass->cpu_data();

        vector<float> glass_vec = {glass_ptr[0], glass_ptr[1]};
        auto glass_value = std::max_element(glass_vec.begin(), glass_vec.end());
        int glass_index = std::distance(glass_vec.begin(), glass_value);

        if (*glass_value > attr_thres) {
            if (glass_index == 1)
                personInfo.glass_info = EM_GLASS_COMMON;
            else if (glass_index == 0)
                personInfo.glass_info = EM_GLASS_FALSE;

            personInfo.glass_score = *glass_value;
        } else {
            personInfo.glass_info = EM_GLASS_UNKNOWN;
        }


    }

    personInfos.push_back(personInfo);
}

void face_interface::predict_facefields(cv::Mat &input_head_img, cv::Mat &input_body_img, PersonInfo &personInfo,
                                        float attr_thres) {
    std::vector<cv::Mat> pkgs;
    pkgs.push_back(input_head_img);

    std::vector<cv::Mat> body_pkgs;
    body_pkgs.push_back(input_body_img);

    std::vector<PersonInfo> personInfos;

    if (personInfo.head_type == EM_PERSON_HEAD_FRONT) {

        personInfos.clear();
        predict_facefields(pkgs, body_pkgs, personInfos, gender_net_, "gender", attr_thres);
        personInfo.gender_info = personInfos[0].gender_info;
        personInfo.gender_score = personInfos[0].gender_score;

        personInfos.clear();
        predict_facefields(pkgs, body_pkgs, personInfos, age_net_, "age", attr_thres);
        personInfo.age_info = personInfos[0].age_info;
        personInfo.age_value = personInfos[0].age_value;

        personInfos.clear();
        predict_facefields(pkgs, body_pkgs, personInfos, net_, "common", attr_thres);
        personInfo.glass_info = personInfos[0].glass_info;
        personInfo.fringe_info = personInfos[0].fringe_info;
        personInfo.mask_info = personInfos[0].mask_info;
        personInfo.hat_info = personInfos[0].hat_info;
        personInfo.bare_info = personInfos[0].bare_info;

//        cv::rectangle(input_body_img, personInfo.facerct, cv::Scalar(0, 0, 255), 2, 8);
//        cv::imshow("img", input_body_img);
//        cv::waitKey(0);

    } else if (personInfo.head_type == EM_PERSON_HEAD_UNKNOWN) {
        personInfo.age_value = EM_AGE_UNKNOWN;
        personInfo.hat_info = EM_HAT_UNKNOWN;
        personInfo.bare_info = EM_BARE_UNKNOWN;

        personInfo.gender_info = EM_GENDER_UNKNOWN;
        personInfo.age_info = EM_AGE_UNKNOWN;
        personInfo.glass_info = EM_GLASS_UNKNOWN;
        personInfo.fringe_info = EM_FRINGE_UNKNOWN;
        personInfo.mask_info = EM_MASK_UNKNOWN;
    }

}






