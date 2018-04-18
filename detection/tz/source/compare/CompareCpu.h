#ifndef __COMPARE_CPU_H_
#define __COMPARE_CPU_H_
#include "Compare.h"

class CompareCpu: public Compare
{
public:
    CompareCpu(int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType):Compare(feaitems, type, cmpType){};
    ~CompareCpu(){};
    virtual void Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh=0.0);
    virtual void Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid);
};
#endif
