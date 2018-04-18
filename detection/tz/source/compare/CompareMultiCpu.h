#ifndef __COMPARE_MULTI_CPU_H_
#define __COMPARE_MULTI_CPU_H_
#include "Compare.h"

class CompareMultiCpu: public Compare
{
public:
    CompareMultiCpu(int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType):Compare(feaitems, type, cmpType){};
    ~CompareMultiCpu(){};
    virtual void Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid);
    virtual void Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh=0.0);


public:
    std::mutex m_mutex;
};
#endif
