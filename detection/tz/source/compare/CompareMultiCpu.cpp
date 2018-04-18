/********************************************************************************* 
  *Copyright(C)
  *FileName: CompareMultiCpu.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2018-2-26
  *Description:  compare feature with Multi CPU
  *History:
   Date: 2018-2-26
   Author: Jin
   Modification: No
**********************************************************************************/

#include "./CompareMultiCpu.h"


void CompareMultiCpu::Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid)
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
		memcpy(m_pfs+m_pos*m_feat_len, m_fi, m_feat_len*sizeof(float));
        m_pos=(m_pos + 1) % m_num;
    }

    if (f.size() > 0 && m_pos == 0)
    {
        m_pos = m_num;
    }
}

void ProcessM(std::string thread_name, size_t threadNo, float *pfs,
                               size_t num_per_thread, size_t index, std::vector<float> &scores, std::vector<std::string> &uuid,
                               float thresh, void *pVoid)
{
    std::cout<< "Thread_name:"<<thread_name<<" num_per_thread:"<<num_per_thread<<" index:"<<index<<" thresh:"<<thresh<<std::endl;
    CompareMultiCpu *pCmp = (CompareMultiCpu *)pVoid;
    float *p_res = NULL;
    p_res = (float*)malloc(num_per_thread*sizeof(float));
    if (NULL == p_res)
    {
        std::cout<< "[ERROR]: ProcessM res malloc error!" <<std::endl;
    }
    memset(p_res, 0, num_per_thread*sizeof(float));

    float alpha=1.f;
    float beta=0.f;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_per_thread, pCmp->m_feat_len, alpha, pfs, pCmp->m_feat_len, pCmp->m_fi, 1, beta, p_res, 1);

    size_t len = num_per_thread;
    std::unique_lock<std::mutex> lock(pCmp->m_mutex);
    for (size_t i = 0; i < len; ++i)
    {
        if (!pCmp->isempty_[index+i])
        {
            if (p_res[i] >= thresh)
            {
                scores.push_back(p_res[i]);
                uuid.push_back(pCmp->uuid_[index+i]);
            }
        }
//        else
//        {
//            scores.push_back(-1.f);
//            uuid.push_back(pCmp->uuid_[index+i]);
//        }
    }

    if (p_res != NULL)
    {
        free(p_res);
    }
}

void CompareMultiCpu::Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh)
{
    std::cout<< "Compare Process: MultiCPU" << std::endl;
    scores.clear();
    if (feat.empty())
    {
        for (size_t i=0; i<m_pos; i++)
        {
            scores.push_back(-1.f);
        }
        uuid=uuid_;
        return;
    }

    std::vector<float> f;
    Parse(feat, f);
    CHECK_EQ(m_feat_len, f.size()) << "feat size not right when compare!";

    float sum=0.f;
    memset(m_fi, 0, m_feat_len*sizeof(float));
    for (size_t j=0; j<m_feat_len; j++)
    {
        sum+=f[j]*f[j];
    }
    for (size_t j=0; j<m_feat_len; j++)
    {
        m_fi[j]=f[j]/sqrt(std::max(sum, 1e-6f));
    }

    std::vector<std::future<void>> item_result;
    size_t thread_count = 10;
    size_t last_num = 0;
    size_t index = 0;
    size_t num_per_thread = m_pos/thread_count;
    bool bEnd = false;
    if (m_pos%thread_count != 0)
    {
        last_num = num_per_thread+m_pos%thread_count;
        bEnd = true;
    }

    std::string thread_name;
    for (size_t thread_no = 0; thread_no < thread_count; ++thread_no)
    {
        thread_name = "Comnpare Thread " + std::to_string(thread_no+1);
        index = thread_no*num_per_thread;
        if (true == bEnd && thread_no == thread_count-1)
        {
            item_result.push_back(std::async(std::launch::async, ProcessM, thread_name, thread_no,
                                             m_pfs+thread_no*num_per_thread*m_feat_len, last_num, index,
                                             std::ref(scores), std::ref(uuid), thresh, (void*)this));
        }
        else
        {
            item_result.push_back(std::async(std::launch::async, ProcessM, thread_name, thread_no,
                                             m_pfs+thread_no*num_per_thread*m_feat_len, num_per_thread, index,
                                             std::ref(scores), std::ref(uuid), thresh, (void*)this));
        }
    }

    for (size_t i = 0; i < thread_count; ++i)
    {
        item_result[i].wait();
    }
}
