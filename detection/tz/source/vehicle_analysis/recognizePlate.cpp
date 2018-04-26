/********************************************************************************* 
  *Copyright(C)
  *FileName: recognizePlate.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2017-12-26
  *Description:  recv image mat and recognize plate
  *History:
   Date: 2017-12-26
   Author: Jin
   Modification: No
**********************************************************************************/

#include "recognizePlate.hpp"
#define BLANK_LABEL_INDEX (68)
#define TIME_STEP (60)

#define CHEPAI_TABLE_SIZE 68
string plate_tab[CHEPAI_TABLE_SIZE+1] =
{
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "京",
    "沪",
    "津",
    "渝",
    "黑",
    "吉",
    "辽",
    "蒙",
    "冀",
    "新",
    "甘",
    "青",
    "陕",
    "宁",
    "豫",
    "鲁",
    "晋",
    "皖",
    "鄂",
    "湘",
    "苏",
    "川",
    "贵",
    "云",
    "桂",
    "藏",
    "浙",
    "赣",
    "粤",
    "闽",
    "琼",
    "挂",
    "学",
    "警",
    " ",
};


#define NUM_COLOR_SIZE 5
string color_tab[NUM_COLOR_SIZE+1]={"蓝","绿","白","黑","黄"};

Classifier::Classifier(const string& model_file, const string& trained_file, const string& mean_file)
{
    bFirst = true;
	#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
	#else
    Caffe::set_mode(Caffe::GPU);
	#endif

    net_.reset(new Net<float>(model_file, TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    //net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    if("" != mean_file)
    {
        use_mean_ = true;
        SetMean(mean_file);
    }
    else
    {
        use_mean_ = false;
    }
}

void Classifier::SetMean(const string& mean_file)
{
    std::vector<cv::Mat> channels;
    BlobProto blob_proto;
    Blob<float> mean_blob;

    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i)
	{
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    cv::Mat mean;
    cv::merge(channels, mean);
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}


const float* Classifier::Predict(const cv::Mat& img, const string& type)
{
    const float *output = NULL;

    if (bFirst == true)
    {
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
        net_->Reshape();
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);
        Preprocess(img, &input_channels);
        net_->Forward();
    }

    if (type == "plate")
    {
        boost::shared_ptr<Blob<float>> out_blob = net_->blob_by_name("premuted_fc");
        output = out_blob->cpu_data();
    }
    else if (type == "color")
    {
        boost::shared_ptr<Blob<float>> out_blob = net_->blob_by_name("fc_color_1");
        output = out_blob->cpu_data();
    }
    else
    {
        std::cout << "error: nomatched type!!!" << std::endl;
        return NULL;
    }

    bFirst = false;

    return output;
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) 
	{
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;

    if(use_mean_)
        cv::subtract(sample_float, mean_, sample_normalized);
    else
        sample_normalized = sample_float.clone();

    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

int Classifier::RecognizePlate(cv::Mat& img, RESULT_PLATE& output)
{
    string result;
    string result_raw;
    const float *output_plate = NULL;
    const float *output_color = NULL;
    int blank_label = BLANK_LABEL_INDEX;
    int prev_label = blank_label;
    int time_step = TIME_STEP;
    int alphabet_size = CHEPAI_TABLE_SIZE+1;
    int color_size = NUM_COLOR_SIZE;

    bFirst = true;
    int64 t1 = cvGetTickCount();
    output_plate = Predict(img, "plate");
    if (NULL == output_plate)
    {
        return -1;
    }

    output_color = Predict(img, "color");
    if (NULL == output_color)
    {
        return -1;
    }

    int64 t2 = cvGetTickCount();
    //std::cout << "recognize time cost:" << (t2-t1)/cvGetTickFrequency() / 1000000 << "s" << std::endl;

    //plate
    for (int i = 0; i < time_step; ++i)
    {
        const float* lin = output_plate + i * alphabet_size;
        int predict_label = std::max_element(lin, lin + alphabet_size) - lin;

        if (predict_label != blank_label && predict_label != prev_label)
        {
            result = result + plate_tab[predict_label];
        }

        result_raw = result_raw + plate_tab[predict_label];
        prev_label = predict_label;
    }

    //color
    int predict_color_label = std::max_element(output_color, output_color + color_size) - output_color;

    //std::cout << "color: " << color_tab[predict_color_label] << std::endl;
    //std::cout << "result: " << result << std::endl;

    output.plateNum = result;
    output.color = color_tab[predict_color_label];

    return 0;
}


