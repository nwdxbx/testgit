#include "person_re_id_v2.hpp"
#define RESIZE_WITH 100
#define RESIZE_HEIGHT 200

Person_Re_ID_v2 &Person_Re_ID_v2::ins()
{
    static thread_local Person_Re_ID_v2 obj("./models/pedestrain_p", "./models/pedestrain_c",0);
    return obj;
}

Person_Re_ID_v2::Person_Re_ID_v2(const std::string &net_file,
                           const std::string &model_file, int gpuid)
                       : netname_(net_file),
                       modelname_(model_file)
{
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(gpuid);
    net_.reset(new caffe::Net<float>(net_file, caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    net_->CopyTrainedLayersFrom(model_file);

    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();

    color_map_.insert(std::pair<int,int>(0,0));
    color_map_.insert(std::pair<int,int>(1,1));
    color_map_.insert(std::pair<int,int>(2,2));
    color_map_.insert(std::pair<int,int>(3,3));
    color_map_.insert(std::pair<int,int>(4,8));
    color_map_.insert(std::pair<int,int>(5,10));
    color_map_.insert(std::pair<int,int>(6,100));
}
void Person_Re_ID_v2::preprocess(cv::Mat &img, cv::Mat &resimg)
{
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
}
void Person_Re_ID_v2::process(std::vector< cv::Mat> &imgin, std::vector<Person_Re_ID_det_res> &result)
{
    //std::cout << "\t\tthe size of the network is (" << w_ << "x" << h_ << ")" << std::endl;

    auto &input_layer=net_->input_blobs()[0];
    input_layer->Reshape({(int)imgin.size(), input_layer->shape(1),
            input_layer->shape(2), input_layer->shape(3)});
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();

    std::vector<std::vector<cv::Mat>>  input_channels;
    input_channels.resize(input_layer->shape()[0]);
    for(int i=0;i<input_layer->shape()[0];i++)
    {
        for (int j=0; j<input_layer->shape()[1]; j++)
        {
            cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
            input_channels[i].push_back(channel);
            input_data += input_layer->width()*input_layer->height();
        }

        cv::Mat x=imgin[i].clone();
        if(x.empty())
            x=cv::Mat(w_,h_,CV_8UC3, cv::Scalar::all(0));
        else
            cv::resize(x, x, cv::Size(w_, h_), (0, 0), (0, 0), cv::INTER_CUBIC);
            //cv::resize(x, x, cv::Size(w_, h_));

        if (x.channels()==1)
            cv::cvtColor(x, x, cv::COLOR_GRAY2BGR);
        cv::Mat floatImg, floatImg1;
        x.convertTo(floatImg1, CV_32FC3);
        preprocess(floatImg1, floatImg);
        cv::split(floatImg, input_channels[i]);

    }
    net_->Forward();

    //get_res(net_->output_blobs(), result);
    auto &up_blob = net_->blob_by_name("loss3/prob/up");
    auto &down_blob = net_->blob_by_name("loss3/prob/down");
    auto &up_color_blob = net_->blob_by_name("loss3/prob/color_up");
    auto &down_color_blob = net_->blob_by_name("loss3/prob/color_down");\
    //auto &bag_blob = net_->blob_by_name("loss3/classifier/bag");
    auto &human_fea_blob = net_->blob_by_name("pool5/7x7_s1");


    const float* up_data = NULL;
    const float* down_data = NULL;
    const float* up_color_data = NULL;
    const float* down_color_data = NULL;
    //const float* bag_data = NULL;
    const float* human_fea_data = NULL;

    up_data=(const float*)up_blob->cpu_data();
    down_data=(const float*)down_blob->cpu_data();
    up_color_data=(const float*)up_color_blob->cpu_data();
    down_color_data=(const float*)down_color_blob->cpu_data();
    //bag_data=(const float*)bag_blob->cpu_data();
    human_fea_data = (const float*)human_fea_blob->cpu_data();

    for(int i = 0; i < up_blob->shape(0); i++)
    {
        //result.resize(up_blob->shape(0));
        Person_Re_ID_det_res st_person;
        //person_fea
        st_person.person_fea.clear();
        int offeset = human_fea_blob->shape(1);
        st_person.person_fea.insert(st_person.person_fea.begin(), human_fea_data, human_fea_data + offeset);
        human_fea_data += offeset;
        //up
        offeset = up_blob->shape(1);
        if(up_data[0] > up_data[1])
        {
            st_person.sleeveLength = 0;
            st_person.sleeveLengthScore = up_data[0];
        }
        else
        {
            st_person.sleeveLength = 1;
            st_person.sleeveLengthScore = up_data[1];
        }
        up_data += offeset;
        //down
        offeset = down_blob->shape(1);
        if(down_data[0] > down_data[1])
        {
            st_person.pantsLength = 0;
            st_person.pantsLengthScore = down_data[0];
        }
        else
        {
            st_person.pantsLength = 1;
            st_person.pantsLengthScore = down_data[1];
        }
        down_data += offeset;
        //up_color
        offeset = up_color_blob->shape(1);
        vector<float> v_up_color(up_color_data, up_color_data + offeset);
        std::vector<int> up_color_Index = sort_indexes(v_up_color);
        st_person.upclsColor = color_map_[up_color_Index[0]];
        st_person.coat_color_score = v_up_color[up_color_Index[0]];
        up_color_data += offeset;
        //down_color
        offeset = down_color_blob->shape(1);
        vector<float> v_down_color(down_color_data, down_color_data + offeset);
        std::vector<int> down_color_Index = sort_indexes(v_down_color);
        st_person.downclsColor = color_map_[down_color_Index[0]];
        st_person.trouser_color_score = v_down_color[up_color_Index[0]];
        down_color_data += offeset;

        result.push_back(st_person);

    }
}
void Person_Re_ID_v2::process(cv::Mat &imgin, Person_Re_ID_det_res &result)
{
    std::vector< cv::Mat > pkgs;
    pkgs.push_back(imgin);
    std::vector<Person_Re_ID_det_res> res;
    process(pkgs, res);
    result.person_fea = res[0].person_fea;
    result.sleeveLength = res[0].sleeveLength;
    result.pantsLength = res[0].pantsLength;
    result.upclsColor = res[0].upclsColor;
    result.downclsColor = res[0].downclsColor;
    result.sleeveLengthScore = res[0].sleeveLengthScore;
    result.pantsLengthScore = res[0].pantsLengthScore;
    result.coat_color_score = res[0].coat_color_score;
    result.trouser_color_score = res[0].trouser_color_score;
}
template <typename T>
std::vector<int> Person_Re_ID_v2:: sort_indexes(const vector<T> &v)
{
    // 初始化索引向量
      vector<int> idx(v.size());
      //使用iota对向量赋0~？的连续值
      iota(idx.begin(), idx.end(), 0);
      // 通过比较v的值对索引idx进行排序
      sort(idx.begin(), idx.end(),
      [&v](int i1, int i2) {return v[i1] > v[i2];});
      //升序[&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
      return idx;
}
