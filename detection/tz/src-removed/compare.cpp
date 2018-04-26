#include "../../include/tuzhenfeature.h"
#include "../../include/tuzhenalginterface.hpp"
#include <cublas_v2.h>
#include <glog/logging.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <math.h>
#include "obj_reid_feat.prototxt.pb.h"
#include "reIDfeature/person_re_id.hpp"
#include "reIDfeature/face_re_id.hpp"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)


class compare
{
public:
    //compare(int deviceID, std::vector<std::string> &feats, std::vector<std::string> &uuid);
    compare(int deviceID, int feaitems, EFeature_Cal_Type type);
    ~compare();
    void process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid);
    void addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid);
private:
    void parse(std::string &feat, std::vector<float> &res);
    const char* cublasGetErrorString(cublasStatus_t error);
    float *fs_;
    size_t len_of_feat_;
    size_t len_of_num_;
    size_t ptr_;
    std::vector<std::string> uuid_;
    std::vector<bool> isempty_;
    cublasHandle_t cublas_handle_;
};

//feaitems: [0,1000000]; objtype: 0 person, 1 face;
compare::compare(int deviceID,  int feaitems, EFeature_Cal_Type type)
{
    CHECK(feaitems > 0);
    ptr_=0;
    len_of_num_=feaitems;
    uuid_.resize(len_of_num_, "EMPTY_ENTRY");
    isempty_.resize(len_of_num_, true);
    
//    if (objtype == 0)
//        len_of_feat_ = config::ins().get_person_feat_len();
//    else
//        len_of_feat_ = config::ins().get_face_feat_len();
    switch (type){
        case EM_FEATURE_CAL_PERSON:{
            len_of_feat_ = 256;
            break;
        }
        case EM_FEATURE_CAL_FACE:{
            len_of_feat_ = 2048;
            break;
        }
    }
    

    CUDA_CHECK(cudaSetDevice(deviceID));

    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }

    CUDA_CHECK(cudaMalloc(&fs_, len_of_num_*len_of_feat_*sizeof(float)));
    CUDA_CHECK(cudaMemset(fs_, 0, len_of_num_*len_of_feat_*sizeof(float)));
}

compare::~compare()
{
    CUDA_CHECK(cudaFree(fs_));
    if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
}

void compare::addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid)
{
    CHECK_EQ(uuid.size(), f.size()) << "feats size and uuid size should be equal!";
    float *fi = new float[len_of_feat_]();
    for (size_t i=0; i<f.size(); i++)
    {
        uuid_[ptr_]=uuid[i];
        std::vector<float> fiv;
        parse(f[i], fiv);
        if (fiv.empty() || f[i].empty())
        {
            isempty_[ptr_]=true;
            fiv.resize(len_of_feat_, 1.);
        }
        else
            isempty_[ptr_]=false;

        CHECK_EQ(fiv.size(), len_of_feat_) << "feat size not right when compare!";

        float sum=0.f;
        for (size_t j=0; j<len_of_feat_; j++)
        {
            sum+=fiv[j]*fiv[j];
        }
        for (size_t j=0; j<len_of_feat_; j++)
        {
            fi[j]=fiv[j]/sqrt(std::max(sum, 1e-6f));
        }
        CUDA_CHECK(cudaMemcpy(fs_+ptr_*len_of_feat_, fi, len_of_feat_*sizeof(float), cudaMemcpyHostToDevice));
        ptr_=(ptr_ + 1) % len_of_num_;
    }
    delete []fi;
}

void compare::process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid)
{
    scores.clear();
    if (feat.empty())
    {
        for (size_t i=0; i<len_of_num_; i++)
        {
            scores.push_back(-1.f);
        }
        uuid=uuid_;
        return;
    }

    std::vector<float> f;
    parse(feat, f);
    CHECK_EQ(len_of_feat_, f.size()) << "feat size not right when compare!";


    float *fi=new float[len_of_feat_]();
    float *fi_d;
    float sum=0.f;
    for (size_t j=0; j<len_of_feat_; j++)
    {
        sum+=f[j]*f[j];
    }
    for (size_t j=0; j<len_of_feat_; j++)
    {
        fi[j]=f[j]/sqrt(std::max(sum, 1e-6f));
    }
    CUDA_CHECK(cudaMalloc(&fi_d, len_of_feat_*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(fi_d, fi, len_of_feat_*sizeof(float), cudaMemcpyHostToDevice));

    float *res=new float [len_of_num_]();
    float *res_d;
    CUDA_CHECK(cudaMalloc(&res_d, len_of_num_*sizeof(float)));
    CUDA_CHECK(cudaMemset(res_d, 0, len_of_num_*sizeof(float)));

    float alpha=1.f;
    float beta=0.f;
    CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T, len_of_feat_, len_of_num_, &alpha, fs_, len_of_feat_, fi_d, 1, &beta, res_d, 1));
    CUDA_CHECK(cudaMemcpy(res, res_d, len_of_num_*sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i=0; i<len_of_num_; i++)
    {
        if (!isempty_[i])
            scores.push_back(res[i]);
        else
            scores.push_back(-1.f);
    }

    CUDA_CHECK(cudaFree(fi_d));
    CUDA_CHECK(cudaFree(res_d));
    delete []fi;
    delete []res;
    uuid=uuid_;
}

void compare::parse(std::string &feat, std::vector<float> &res)
{
    try
    {
        res.clear();
        if (feat.empty())
            return;
        face_feat_entry entry;
        entry.ParseFromString(feat);

        for (size_t i=0; i<entry.reid_face_feat_size(); i++)
        {
            res.push_back(entry.reid_face_feat(i));
        }
    }
    catch(...)
    {
        res.clear();
        return;
    }
}

const char* compare::cublasGetErrorString(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}



int feacompare_open(void **ppvFeaCmpHandle, int deviceID, int feaitems, EFeature_Cal_Type type)
{
    void *pFeatCmpHandle = new class compare(deviceID, feaitems, type);

    if(pFeatCmpHandle == NULL)
        return EM_ERROR_MALLOC;

    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //caffe::Caffe::SetDevice(deviceID);


    //Person_Re_ID::ins();
    //Face_Re_ID::ins();

    *ppvFeaCmpHandle = pFeatCmpHandle;
    return 0;
}



int feacompare_process(void *pvFeaCmpHandle, std::string &feature, std::vector<float> &res, std::vector<std::string> &uuid)
{
    class compare *pCompare = (class compare*)pvFeaCmpHandle;
    pCompare->process(feature, res, uuid);

    return 0;
}

int feacompare_addfeat(void *pvFeaCmpHandle, std::vector<std::string> &fea, std::vector<std::string> &uuid)
{
    class compare *pCompare = (class compare*)pvFeaCmpHandle;
    pCompare->addfeat(fea, uuid);
    return 0;
}

void feacompare_close(void *pvFeaCmpHandle)
{
    class compare *pCompare = (class compare*)pvFeaCmpHandle;
    delete pCompare;
    return;
}


