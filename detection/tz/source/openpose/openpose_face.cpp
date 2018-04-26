#include "openpose_face.hpp"
#include "openposeResize.hpp"
#include "openposeNms.hpp"
#include "property.hpp"


openpose_face::openpose_face(const std::string &modelpath, cv::Size netsz)
{
    netname_=modelpath+property::FACE_PROTOTXT;
    modelname_=modelpath+property::FACE_TRAINED_MODEL;
    outputname_="net_output";
    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(0);
    net_.reset(new caffe::Net<float>{netname_, caffe::TEST});
    net_->CopyTrainedLayersFrom(modelname_);

    netsz.width = netsz.width / 16 * 16;
    netsz.height = netsz.height / 16 * 16;
    net_->blobs()[0]->Reshape({1, 3, netsz.height, netsz.width});
    net_->Reshape();

    res_blob_ = net_->blob_by_name(outputname_);
    
    heatmap_blob_.reset(new caffe::Blob<float>(1,1,1,1));

    kernel_blob_.reset(new caffe::Blob<int>(1,1,1,1));

    peak_blob_.reset(new caffe::Blob<float>(1,1,1,1));

    caffe::Blob<float> *input_layer=net_->input_blobs()[0];

    w_=input_layer->width();
    h_=input_layer->height();
}

void openpose_face::process(cv::Mat &img1, std::vector<cv::Rect2f> &person_face, std::vector<std::vector<cv::Vec3f>> &pts)
{
    cv::Mat img=img1.clone();
    if (img.empty())
        img=cv::Mat(h_, w_, CV_8UC3);
    if (img.channels()!=3)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);


    pts.resize(person_face.size());

    for (size_t id=0; id<person_face.size(); id++)
    {
        cv::Rect2f rect = person_face[id];
        float minfacesz=std::min(rect.width, rect.height);
        if (minfacesz>40)
        {
            float maxfacesz=std::max(rect.width, rect.height);
            const double scale=maxfacesz/std::min(net_input_sz_.width, net_input_sz_.height);
            cv::Mat Mscaling=cv::Mat::eye(2,3,CV_64F);
            Mscaling.at<double>(0,0)=scale;
            Mscaling.at<double>(1,1)=scale;
            Mscaling.at<double>(0,2)=rect.x;
            Mscaling.at<double>(1,2)=rect.y;

            cv::Mat faceImg;
            cv::warpAffine(img, faceImg, Mscaling, net_input_sz_,
                           CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));


            caffe::Blob<float> *input_layer=net_->input_blobs()[0];
            input_layer->Reshape({1, input_layer->shape(1), input_layer->shape(2), input_layer->shape(3)});
            net_->Reshape();

            heatmap_blob_->Reshape({1, res_blob_->shape()[1],
                net_input_sz_.height, net_input_sz_.width});

            kernel_blob_->Reshape({1, res_blob_->shape()[1], 
                net_input_sz_.height, net_input_sz_.width});

            peak_blob_->Reshape({1, (int)property::FACE_MAX_PEAKS, 
                (int)property::FACE_NUMBER_PARTS+1, 3});

            float *input_data=input_layer->mutable_cpu_data();
            std::vector<cv::Mat> channels;
            for (int i=0; i<img.channels(); i++)
            {
                cv::Mat channel(img.rows, img.cols, CV_32FC1);
                channels.push_back(channel);
                input_data+=input_layer->width()*input_layer->height();
            }
            cv::split(img, channels);

            net_->Forward();

            std::array<int, 4> sz1{{heatmap_blob_->shape()[0], heatmap_blob_->shape()[1], 
                heatmap_blob_->shape()[2], heatmap_blob_->shape()[3]}};
            
            std::array<int, 4> sz2{{res_blob_->shape()[0], res_blob_->shape()[1], 
                res_blob_->shape()[2], res_blob_->shape()[3]}};

            std::array<int, 4> sz3{{peak_blob_->shape()[0], peak_blob_->shape()[1], 
                peak_blob_->shape()[2], peak_blob_->shape()[3]}};


            resizeAndMergeGpu(heatmap_blob_->mutable_gpu_data(),
                    res_blob_->mutable_gpu_data(), sz1, sz2);


            nmsGpu(peak_blob_->mutable_gpu_data(), 
                    kernel_blob_->mutable_gpu_data(), 
                    heatmap_blob_->mutable_gpu_data(), 
                    (float)property::FACE_DEFAULT_NMS_THRESHOLD,
                    sz3, sz1);


            pts[id].resize((int)property::FACE_NUMBER_PARTS);
            const auto* facePeaksPtr = peak_blob_->mutable_cpu_data();
            const auto facePeaksOffset = (property::FACE_MAX_PEAKS+1) * 3;

            for (size_t part = 0 ; part < pts[id].size() ; part++)
            {
                // Get max peak
                const int numPeaks = int(facePeaksPtr[facePeaksOffset*part]+0.5f);
                auto maxScore = -1.f;
                auto maxPeak = -1;
                for (auto peak = 0 ; peak < numPeaks ; peak++)
                {
                    const auto xyIndex = facePeaksOffset * part + (1 + peak) * 3;
                    const auto score = facePeaksPtr[xyIndex + 2];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxPeak = peak;
                    }
                }
                // Fill face keypoints
                if (maxPeak >= 0)
                {
                    const auto xyIndex = facePeaksOffset * part + (1 + maxPeak) * 3;
                    const auto x = facePeaksPtr[xyIndex];
                    const auto y = facePeaksPtr[xyIndex + 1];
                    const auto score = facePeaksPtr[xyIndex + 2];
                    pts[id][0] =  (Mscaling.at<double>(0,0) * x + Mscaling.at<double>(0,1) * y + Mscaling.at<double>(0,2))/scale;
                    pts[id][1] =  (Mscaling.at<double>(1,0) * x + Mscaling.at<double>(1,1) * y + Mscaling.at<double>(1,2))/scale;
                    pts[id][2] = score;
                }
            }
        }
    }
}

float openpose_face::getDistance(cv::Vec3f pt1, cv::Vec3f pt2)
{
    float distx=pt1[0]-pt2[0];
    float disty=pt1[1]-pt2[1];
    return std::sqrt(distx*distx+disty*disty);
}

cv::Rect2f openpose_face::getfacefromposekeypoints(const std::vector<std::vector<cv::Vec3f>>& posekeypoints, unsigned int personindex, const unsigned int neck,
        const unsigned int headnose, const unsigned int lear, const unsigned int rear,
        const unsigned int leye, const unsigned int reye, const float threshold)
{
    cv::Point2f pointTopLeft{0.f, 0.f};
    auto faceSize = 0.f;

    auto &kpts=posekeypoints[personindex];
    const auto neckScoreAbove = (kpts[neck][2] > threshold);
    const auto headNoseScoreAbove = (kpts[headnose][2] > threshold);
    const auto lEarScoreAbove = (kpts[lear][2] > threshold);
    const auto rEarScoreAbove = (kpts[rear][2] > threshold);
    const auto lEyeScoreAbove = (kpts[leye][2] > threshold);
    const auto rEyeScoreAbove = (kpts[reye][2] > threshold);

    auto counter = 0;
    // Face and neck given (e.g. MPI)
    if (headnose == lear && lear == rear)
    {
        if (neckScoreAbove && headNoseScoreAbove)
        {
            pointTopLeft.x = kpts[headnose][0];
            pointTopLeft.y = kpts[headnose][1];
            faceSize = 1.33f * getDistance(kpts[neck], kpts[headnose]);
        }
    }
    // Face as average between different body keypoints (e.g. COCO)
    else
    {
        // factor * dist(neck, headNose)
        if (neckScoreAbove && headNoseScoreAbove)
        {
            // If profile (i.e. only 1 eye and ear visible) --> avg(headNose, eye & ear position)
            if ((lEyeScoreAbove) == (lEarScoreAbove)
                    && (rEyeScoreAbove) == (rEarScoreAbove)
                    && (lEyeScoreAbove) != (rEyeScoreAbove))
            {
                if (lEyeScoreAbove)
                {
                    pointTopLeft.x += (kpts[leye][0] + kpts[lear][0] + kpts[headnose][0]) / 3.f;
                    pointTopLeft.y += (kpts[leye][1] + kpts[lear][1] + kpts[headnose][1]) / 3.f;
                    faceSize += 0.85f * (getDistance(kpts[headnose], kpts[leye]) + getDistance(kpts[headnose], kpts[lear]) + getDistance(kpts[neck], kpts[headnose]));
                }
                else // if(lEyeScoreAbove)
                {
                    pointTopLeft.x += (kpts[reye][0] + kpts[rear][0] + kpts[headnose][0]) / 3.f;
                    pointTopLeft.y += (kpts[reye][1] + kpts[rear][1] + kpts[headnose][1]) / 3.f;
                    faceSize += 0.85f * (getDistance(kpts[headnose], kpts[reye]) + getDistance(kpts[headnose], kpts[rear]) + getDistance(kpts[neck], kpts[headnose]));
                }
            }
            // else --> 2 * dist(neck, headNose)
            else
            {
                pointTopLeft.x += (kpts[neck][0] + kpts[headnose][0]) / 2.f;
                pointTopLeft.y += (kpts[neck][1] + kpts[headnose][1]) / 2.f;
                faceSize += 2.f * getDistance(kpts[neck], kpts[headnose]);
            }
            counter++;
        }
        // 3 * dist(lEye, rEye)
        if (lEyeScoreAbove && rEyeScoreAbove)
        {
            pointTopLeft.x += (kpts[leye][0] + kpts[reye][0]) / 2.f;
            pointTopLeft.y += (kpts[leye][1] + kpts[reye][1]) / 2.f;
            faceSize += 3.f * getDistance(kpts[leye], kpts[reye]);
            counter++;
        }
        // 2 * dist(lEar, rEar)
        if (lEarScoreAbove && rEarScoreAbove)
        {
            pointTopLeft.x += (kpts[lear][0] + kpts[rear][0]) / 2.f;
            pointTopLeft.y += (kpts[lear][1] + kpts[rear][1]) / 2.f;
            faceSize += 2.f * getDistance(kpts[lear], kpts[rear]);
            counter++;
        }
        // Average (if counter > 0)
        if (counter > 0)
        {
            pointTopLeft /= (float)counter;
            faceSize /= counter;
        }
    }
    return cv::Rect2f{pointTopLeft.x-faceSize/2, pointTopLeft.y-faceSize/2, faceSize, faceSize};
}
