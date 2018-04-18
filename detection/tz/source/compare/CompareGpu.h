#ifndef __COMPARE_GPU_H_
#define __COMPARE_GPU_H_
#include "Compare.h"


class CompareGpu: public Compare
{
public:
    CompareGpu(int deviceID,  int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType);
    ~CompareGpu();
    virtual void Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh=0.0);
    virtual void Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid);

private:
    const char* cublasGetErrorString(cublasStatus_t error);
    cublasHandle_t cublas_handle_;
};
#endif
