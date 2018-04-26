#include "object_detector.h"
#include <cstdio>
#include <ctime>
#include <fstream>

//int current_gpu_id = 0;
int yoloDetector::max_idx(std::vector<float> &f) {
    float max = f[0];
    int id = 0;
    for (int i = 1; i < f.size(); i++) {
        if (max < f[i]) {
            max = f[i];
            id = i;
        }
    }
    return id;
}

float yoloDetector::iou(det_class &det1, det_class &det2) {
    float area1 = det1.w * det1.h;
    float area2 = det2.w * det2.h;

    float over_w = std::max(0.f, (det1.w + det2.w) -
                                 (std::max(det1.x + det1.w, det2.x + det2.w) - std::min(det1.x, det2.x)));
    float over_h = std::max(0.f, (det1.h + det2.h) -
                                 (std::max(det1.y + det1.h, det2.y + det2.h) - std::min(det1.y, det2.y)));

    float over_area = over_w * over_h;

    return over_area / std::min(area1, area2);
}

void yoloDetector::non_maxima_suppression(std::vector<det_class> &det, float fth) {
    for (int i = 0; i < det.size(); i++) {
        if (det[i].id == -1)
            continue;
        for (int j = i + 1; j < det.size(); j++) {
            if (det[i].id == det[j].id && iou(det[i], det[j]) > fth)
                det[j].id = -1;
        }
    }
}

yoloDetector::yoloDetector(char *cfgfile, char *weightfile, int gpu_id,float nms, float thresh, float hier_thresh) {
    _nms = nms;
    _thresh = thresh;
    _hier_thresh = hier_thresh;
    _cfgfile = cfgfile;
    _weightfile = weightfile;

    cuda_set_device(gpu_id);
    //current_gpu_id++;
    _net = parse_network_cfg(_cfgfile);
    if (_weightfile) {
        load_weights(_net, _weightfile);
    }
    set_batch_network(_net, 1);
    _l = _net->layers[_net->n - 1];
}

yoloDetector::~yoloDetector() {

}

void yoloDetector::detector(cv::Mat &img, std::vector<det_class> &det) {
    srand(2222222);
    clock_t time;

    cv::Mat sample_single;
    cv::Mat m = img.clone();
    cv::resize(m, m, cv::Size(_net->w, _net->h));
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
    m.convertTo(sample_single, CV_32FC3);        //转浮点型
    sample_single = sample_single / 255.0;

    size_t len = _net->h * _net->w * 3;
    float *X = new float[len];
    std::vector<cv::Mat> *input_channels = new std::vector<cv::Mat>;
    for (int i = 0; i < m.channels(); ++i) {
        float *XX = X + i * _net->h * _net->w;
        cv::Mat channel(_net->h, _net->w, CV_32FC1, XX);
        input_channels->push_back(channel);

    }
    cv::split(sample_single, *input_channels);

    network_predict(_net, X);
    delete[]X;
//    time=clock();
//    printf("Predicted in %f seconds.\n", sec(clock()-time));

    int map_w = _l.w;
    int map_h = _l.h;
    int num_class = _l.classes;
    int feature_len = _l.classes + _l.coords + 1;
    float *predictions = _l.output;
    for (int i = 0; i < _l.h; i++) {
        for (int j = 0; j < _l.w; j++) {
            for (int k = 0; k < _l.n; k++) {
                std::vector<float> fclass;
                for (int id = 0; id < num_class; id++) {
                    fclass.push_back(
                            predictions[map_h * map_w * feature_len * k + map_h * map_w * (5 + id) + map_w * i + j]);
                }

                det_class obj;
                obj.x = (j + predictions[map_h * map_w * feature_len * k + map_h * map_w * 0 + map_w * i + j]) / map_w *
                        img.cols;
                obj.y = (i + predictions[map_h * map_w * feature_len * k + map_h * map_w * 1 + map_w * i + j]) / map_h *
                        img.rows;
                obj.w = exp(predictions[map_h * map_w * feature_len * k + map_h * map_w * 2 + map_w * i + j]) *
                        _l.biases[2 * k] / map_w * img.cols;
                obj.h = exp(predictions[map_h * map_w * feature_len * k + map_h * map_w * 3 + map_w * i + j]) *
                        _l.biases[2 * k + 1] / map_h * img.rows;
                obj.id = max_idx(fclass);
                obj.score = fclass[obj.id] *
                            predictions[map_h * map_w * feature_len * k + map_h * map_w * 4 + map_w * i + j];
                if (obj.score > _thresh) {
                    det.push_back(obj);
                }
            }
        }
    }
    std::sort(det.begin(), det.end(), [](const det_class &a, const det_class &b) { return a.score > b.score; });
    non_maxima_suppression(det, _nms);
}

void yoloDetector::process(cv::Mat &img, std::vector<det_class> &out) {
    std::vector<det_class> det;
    detector(img, det);
    for (int i = 0; i < det.size(); i++) {
        if (det[i].id == -1)
            continue;

        det[i].x -= det[i].w / 2;
        det[i].y -= det[i].h / 2;
        if (det[i].x < 0) det[i].x = 0;
        if (det[i].y < 0) det[i].y = 0;
        if (det[i].x + det[i].w > img.cols - 1) det[i].w = img.cols-det[i].x-1;
        if (det[i].y + det[i].h > img.rows - 1) det[i].h = img.rows-det[i].y-1;
        out.push_back(det[i]);


        // cv::Rect r;
        // r.x = det[i].x;
        // r.y = det[i].y;
        // r.width = det[i].w;
        // r.height = det[i].h;

        // r.x -= r.width / 2;
        // r.y -= r.height / 2;
        // if (r.x < 0) r.x = 0;
        // if (r.y < 0) r.y = 0;
        // out.push_back({r, det[i]});
    }
}

void yoloDetector::humanprocess(cv::Mat &img, std::vector<det_class> &out)
{
    std::vector<det_class> det;
    detector(img, det);
    for (int i = 0; i < det.size(); i++) {
        if (det[i].id != 0)
            continue;

        det[i].x -= det[i].w / 2;
        det[i].y -= det[i].h / 2;
        if (det[i].x < 0) det[i].x = 0;
        if (det[i].y < 0) det[i].y = 0;
        if (det[i].x + det[i].w > img.cols - 1) det[i].w = img.cols-det[i].x-1;
        if (det[i].y + det[i].h > img.rows - 1) det[i].h = img.rows-det[i].y-1;
        out.push_back(det[i]);
    }
}

