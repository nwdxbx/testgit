#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace caffe;
using boost::shared_ptr;
using std::string;
using std::vector;
using std::cout;
using std::endl;

#define BLANK_LABEL_INDEX (68)

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
    " "
};


#define NUM_COLOR_SIZE 5
string color_tab[NUM_COLOR_SIZE+1]={"蓝","绿","白","黑","黄"};

class Classifier {
public:
    Classifier(const string& model_file, const string& trained_file, const string& mean_file);
    const float* Predict(const cv::Mat& img, const string& type);

private:
    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    bool use_mean_;
    cv::Mat mean_;
    std::vector<string> labels_;

public:
    bool bFirst;
};

Classifier::Classifier(const string& model_file, const string& trained_file, const string& mean_file)
{
    bFirst = true;
    Caffe::set_mode(Caffe::GPU);

    net_.reset(new Net<float>(model_file, TEST));
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
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    cv::Mat mean;
    cv::merge(channels, mean);

    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

//const float* Classifier::Predict(const cv::Mat& img)
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
        shared_ptr<Blob<float>> out_blob = net_->blob_by_name("premuted_fc");
        output = out_blob->cpu_data();
    }
    else if (type == "color")
    {
        shared_ptr<Blob<float>> out_blob = net_->blob_by_name("fc_color_1");
        output = out_blob->cpu_data();
    }
    else
    {
        cout << "error: nomatched type!!!" << endl;
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
    for (int i = 0; i < input_layer->channels(); ++i) {
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

struct Info
{
    int idx;
    float score;
};

bool sort_score(Info i1, Info i2)
{
    return i1.score > i2.score;
}

int main(int argc, char** argv)
{
    string model_file   = "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/lstm_ocr/deploy.prototxt";
    string trained_file = "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/lstm_ocr/mylstm_iter_f_14200.caffemodel";
    string mean_file = "";
    std::ifstream list("/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/lstm_ocr/list.txt");
    int blank_label = BLANK_LABEL_INDEX;
    int time_step = 60;
    int alphabet_size = 69;
    int color_size = NUM_COLOR_SIZE;
    int num = 0;
    int failed_num = 0;
    float accuracy = 0.0;
    int failed_num_color = 0;
    float accuracy_color = 0.0;
    string failed_pic_path = "";

    Classifier classifier(model_file, trained_file, mean_file);

    string file;
    while(list >> file)
    {
        cout << "file_name= " << file <<endl;
        failed_pic_path = file;
        //failed_pic_path = failed_pic_path.replace(failed_pic_path.find("test"), 4, "fail");
        //cout << "failed_pic_path=" << failed_pic_path << endl;
        classifier.bFirst = true;
        cv::Mat img = cv::imread(file);
        int64 t1 = cvGetTickCount();
        //const float *output = classifier.Predict(img);
        const float *output_plate = classifier.Predict(img, "plate");
        const float *output_color = classifier.Predict(img, "color");

        int64 t2 = cvGetTickCount();
        cout << "time cost:" << (t2-t1)/cvGetTickFrequency() / 1000000 << "s" << endl;

        int prev_label = blank_label;
        string result, result_raw;
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

        cout << "color: " << color_tab[predict_color_label] << endl;
        cout << "result: " << result << endl;
        //cout << "result_raw: " << result_raw << endl;
        //filter
        std::size_t pos = file.find_last_of("/");
        string pic_name(file.substr(pos+1));
        std::vector<string> plate_vec;
        boost::split(plate_vec, pic_name, boost::is_any_of("_"), boost::token_compress_on);
        string plate_num = plate_vec[1];
        int plate_color = atoi(plate_vec[2].c_str());
        cout << "plate_num:" << plate_num << endl;
        cout << "plate_color:" << plate_color << endl;
        num++;

        if (plate_num != result)
        {
            failed_num++;
            cout << "plate_num:" << plate_num << endl;
            cout << "result___:" << result <<endl;
            cout << "failed recognition num:" << failed_num << endl;
            cout << "total  recognition num:" << num << endl;
            cout << endl;
            //cv::imwrite(failed_pic_path, img);
        }

        if (plate_color != predict_color_label)
        {
            failed_num_color++;
            cout << "color_org: " << color_tab[plate_color] << endl;
            cout << "color_pre: " << color_tab[predict_color_label] << endl;
            cout << "failed color num:" << failed_num_color << endl;
            cout << "total  color num:" << num << endl;
            cout << endl;
            //cv::imwrite(failed_pic_path, img);
        }

        imshow("img", img);
        cv::waitKey(0);
    }

    accuracy = ((num-failed_num)/(float)num);
    cout << "accuracy:" << accuracy << endl;


    accuracy_color = ((num-failed_num_color)/(float)num);
    cout << "accuracy_color:" << accuracy_color << endl;

    return 0;
}
