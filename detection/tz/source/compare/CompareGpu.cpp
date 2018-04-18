/********************************************************************************* 
  *Copyright(C)
  *FileName: CompareGpu.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2018-1-4
  *Description:  compare feature with GPU
  *History:
   Date: 2018-1-4
   Author: Jin
   Modification: YES
**********************************************************************************/

#include "./CompareGpu.h"


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


CompareGpu::CompareGpu(int deviceID,  int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType):Compare(feaitems, type, cmpType)
{
    CUDA_CHECK(cudaSetDevice(deviceID));
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }

    CUDA_CHECK(cudaMalloc(&m_pfs, m_num*m_feat_len*sizeof(float)));
    CUDA_CHECK(cudaMemset(m_pfs, 0, m_num*m_feat_len*sizeof(float)));
}


CompareGpu::~CompareGpu()
{
    CUDA_CHECK(cudaFree(m_pfs));
    if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
}

void CompareGpu::Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid)
{
    CHECK_EQ(uuid.size(), f.size()) << "feats size and uuid size should be equal!";
	memset(m_fi, 0, m_feat_len*sizeof(float));
    for (size_t i=0; i<f.size(); i++)
    {
        uuid_[m_pos]=uuid[i];
        std::vector<float> fiv;
        Parse(f[i], fiv);
        if (fiv.empty() || f[i].empty())
        {
            isempty_[m_pos]=true;
            fiv.resize(m_feat_len, 1.);
        }
        else
            isempty_[m_pos]=false;

        CHECK_EQ(fiv.size(), m_feat_len) << "feat size not right when compare!";

        float sum=0.f;
        for (size_t j=0; j<m_feat_len; j++)
        {
            sum+=fiv[j]*fiv[j];
        }
        for (size_t j=0; j<m_feat_len; j++)
        {
            m_fi[j]=fiv[j]/sqrt(std::max(sum, 1e-6f));
        }
        CUDA_CHECK(cudaMemcpy(m_pfs+m_pos*m_feat_len, m_fi, m_feat_len*sizeof(float), cudaMemcpyHostToDevice));
        m_pos=(m_pos + 1) % m_num;
    }
}

void CompareGpu::Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh)
{
	std::cout<< "Compare Process: GPU" << std::endl;
    scores.clear();
    if (feat.empty())
    {
        for (size_t i=0; i<m_num; i++)
        {
            scores.push_back(-1.f);
        }
        uuid=uuid_;
        return;
    }

    std::vector<float> f;
    Parse(feat, f);
    CHECK_EQ(m_feat_len, f.size()) << "feat size not right when compare!";

    float *fi_d;
    float sum=0.f;
	memset(m_fi, 0, m_feat_len*sizeof(float));
	memset(m_res, 0, m_num*sizeof(float));
	
    for (size_t j=0; j<m_feat_len; j++)
    {
        sum+=f[j]*f[j];
    }
    for (size_t j=0; j<m_feat_len; j++)
    {
        m_fi[j]=f[j]/sqrt(std::max(sum, 1e-6f));
    }
    CUDA_CHECK(cudaMalloc(&fi_d, m_feat_len*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(fi_d, m_fi, m_feat_len*sizeof(float), cudaMemcpyHostToDevice));

    float *res_d;
    CUDA_CHECK(cudaMalloc(&res_d, m_num*sizeof(float)));
    CUDA_CHECK(cudaMemset(res_d, 0, m_num*sizeof(float)));

    float alpha=1.f;
    float beta=0.f;
    std::cout << "m_num:" << m_num << endl <<  "m_feat_len" << m_feat_len<<endl;
    CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T, m_feat_len, m_num, &alpha, m_pfs, m_feat_len, fi_d, 1, &beta, res_d, 1));
    CUDA_CHECK(cudaMemcpy(m_res, res_d, m_num*sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i=0; i<m_num; i++)
    {
        if (!isempty_[i])
        {
            if (m_res[i] >= thresh)
            {
                scores.push_back(m_res[i]);
                uuid.push_back(uuid_[i]);
            }
        }
//        else
//        {
//            scores.push_back(-1.f);
//        }
    }

    CUDA_CHECK(cudaFree(fi_d));
    CUDA_CHECK(cudaFree(res_d));
}

const char* CompareGpu::cublasGetErrorString(cublasStatus_t error)
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

