#ifndef __COMPARE_H_
#define __COMPARE_H_
#include "./tuzhenfeature.h"
#include "./tuzhenalginterface.hpp"
#include <cblas.h>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <math.h>
#include "../source/compare/obj_reid_feat.prototxt.pb.h"
#include "../source/reIDfeature/person_re_id.hpp"
#include "../source/reIDfeature/face_re_id.hpp"
#include <iostream>
#include <future>
#include <mutex>


class Compare
{
public:
    Compare(int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType);
    virtual ~Compare();
    virtual void Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh=0.0) = 0;
    virtual void Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid) = 0;
    void Parse(std::string &feat, std::vector<float> &res);

public:
	float *m_fi;
	float *m_res;
	float *m_pfs;
	size_t m_feat_len;
	size_t m_num;
	size_t m_pos;
	std::vector<std::string> uuid_;
	std::vector<bool> isempty_;
};
#endif
