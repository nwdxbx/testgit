
#include "car_type_color.h"

template <typename T>
std::vector<int> sort_indexes(const std::vector<T> &v)
{
    // 初始化索引向量
      std::vector<int> idx(v.size());
      //使用iota对向量赋0~？的连续值
      iota(idx.begin(), idx.end(), 0);
      // 通过比较v的值对索引idx进行排序
      sort(idx.begin(), idx.end(),
           [&v](int i1, int i2) {return v[i1] > v[i2];});
      //   升序
//      [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
      return idx;
}

car_type_color &car_type_color::ins()
{

    static thread_local car_type_color type_obj;
    return type_obj;
}


car_type_color::car_type_color()
{

    std::string color_netName_ = "./models/car_color_11_p";
    std::string color_modelName_="./models/car_color_11_c";

    std::string brand_netName_ = "./models/chejian_2249_p";
    std::string brand_modelName_="./models/chejian_2249_c";

//    std::string brand_netName_ = "/hard_disk2/Receive_files/xls/git/tuzhen/QT/bin/未加密/0313/chejian_2294.prototxt";
//    std::string brand_modelName_="/hard_disk2/Receive_files/xls/git/tuzhen/QT/bin/未加密/0313/_iter_151400.caffemodel";

    vehicle_brand_net_.reset(new caffe::Net<float>(brand_netName_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    //vehicle_brand_net_.reset(new caffe::Net<float>(brand_netName_,caffe::TEST));
    vehicle_brand_net_->CopyTrainedLayersFrom(brand_modelName_);

    vehicle_color_net_.reset(new caffe::Net<float>(color_netName_,caffe::TEST,"B743382C96DB85858FF65AE97DF98970","802030C17462D3E3"));
    vehicle_color_net_->CopyTrainedLayersFrom(color_modelName_);

    auto &input_layer=vehicle_brand_net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();

    //std::string map_type_path="/hard_disk2/Receive_files/xls/git/tuzhen/QT/bin/未加密/0313/head_2294.label";
    std::string map_type_path="./models/head_2294.label";
    std::string map_color_path="./models/color_11.tag";
    std::vector<std::string> map_path;
    map_path.push_back(map_type_path);
    map_path.push_back(map_color_path);


    //std::string map_type_path=map_path[0];
    std::ifstream lableFile(map_type_path);
    std::string labelStr;
    int labelCount=0;

    while(std::getline(lableFile,labelStr))
    {
        char* strcopy = new char[labelStr.length() + 1];
        strcpy(strcopy,labelStr.c_str());
        char *word = strtok(strcopy," ");
        while(word=strtok(nullptr," "))
        {
            std::string s(word);
            int nPos = labelStr.find(" ");
            std::string newStr = labelStr.substr(0,nPos);
            labelCount = atoi(s.c_str());
            labelMap[labelCount]=newStr;
            break;
        }
    }

    /******* car color id******/
    //std::string map_color_path=map_path[1];
    std::ifstream colorFile(map_color_path);
    std::string colorStr;
    int colorCount=0;
    while(std::getline(colorFile,colorStr))
    {
        char* strcopy = new char[colorStr.length() + 1];
        strcpy(strcopy,colorStr.c_str());
        char *word = strtok(strcopy," ");
        while(word=strtok(nullptr," "))
        {
            std::string s(word);
            int nPos = colorStr.find(" ");
            std::string newStr = colorStr.substr(0,nPos);
            colorCount = atoi(s.c_str());
            colorMap[colorCount]=newStr;
            break;
        }
    }


}

void car_type_color::preprocess(cv::Mat &img, cv::Mat &resimg)
{    
    resimg=(img-cv::Scalar(127.5, 127.5, 127.5))*0.0078125;
}

void car_type_color::process(cv::Mat &imgin, std::vector<float> &output,VehicleDetectionType type)
{
    //std::cout << "\t\tthe size of the network is (" << w_ << "x" << h_ << ")" << std::endl;
    boost::shared_ptr<caffe::Net<float> > net_;
    if(type == VEHICLE_BRAND){
        net_ = vehicle_brand_net_;
    }else{
        net_ = vehicle_color_net_;
    }
    auto &input_layer=net_->input_blobs()[0];
    w_=input_layer->width();
    h_=input_layer->height();
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();
    std::vector<cv::Mat> input_channels;
 //   input_channels.resize(input_layer->shape()[0]);
    for (int j=0; j<input_layer->shape()[1]; j++)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->width()*input_layer->height();
    }

    cv::Mat x=imgin.clone();
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
    cv::split(floatImg, input_channels);

    net_->Forward();
    const std::vector<caffe::Blob<float>*> net_res=net_->output_blobs();
    const float *pData = net_res[0]->cpu_data();
    output.insert(output.begin(), pData, pData+(net_res[0]->shape(1)));
//    caffe::shared_ptr<caffe::Blob<float> > res = net_->blob_by_name("loss3/classifier/18/256");
//    const float *resk = res->cpu_data();
//    output.insert(output.begin(),resk,resk+res->shape(1));

}

void car_type_color::car_id(cv::Mat &img,std::vector<std::string> &vecResult,float type_thresh,float color_thresh)
{

    cv::Mat newSrc;
    cv::Scalar value;
    cv::Mat new_img = img.clone();
    if(new_img.rows>new_img.cols)
    {
        float diff = new_img.rows-new_img.cols;
        value=cv::Scalar(128,128,128);
        cv::copyMakeBorder(new_img,newSrc,0,0,diff/2.0,diff-diff/2.0,cv::BORDER_CONSTANT,value);
    }
    else
    {
        float diff = new_img.cols-new_img.rows;
        value=cv::Scalar(128,128,128);
        cv::copyMakeBorder(new_img,newSrc,diff/2.0,diff-diff/2.0,0,0,cv::BORDER_CONSTANT,value);
    }

//    cv::imshow("img", new_img);
//    cv::waitKey(0);
//    cv::imshow("pad", newSrc);
//    cv::waitKey(0);


    /******* car type id******/
     std::vector<float> output;
     car_type_color::ins().process(newSrc,output,VEHICLE_BRAND);
     std::vector<int> vecIndex = sort_indexes(output);
     if(output[vecIndex[0]]>type_thresh)
     {
        vecResult.push_back(labelMap[vecIndex[0]]);
     }
     else
     {
        vecResult.push_back("Unknow");
     }

    /******* car color id******/

      std::vector<float> colorOutput;
      car_type_color::ins().process(newSrc,colorOutput,VEHICLE_COLOR);
      std::vector<int> colorIndex = sort_indexes(colorOutput);
      if(colorOutput[colorIndex[0]]>color_thresh)
      {
         vecResult.push_back(colorMap[colorIndex[0]]);
      }
      else
      {
         vecResult.push_back("Unknow");
      }
}
