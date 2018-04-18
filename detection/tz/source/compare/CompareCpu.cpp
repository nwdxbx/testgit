/********************************************************************************* 
  *Copyright(C)
  *FileName: CompareApi.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2018-1-4
  *Description:  compare feature with CPU
  *History:
   Date: 2018-1-4
   Author: Jin
   Modification: YES
**********************************************************************************/

#include "./CompareCpu.h"


void CompareCpu::Addfeat(std::vector<std::string> &f, std::vector<std::string> &uuid)
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
}

void CompareCpu::Process(std::string &feat, std::vector<float> &scores, std::vector<std::string> &uuid,float thresh)
{
	std::cout<< "Compare Process: CPU" << std::endl;
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

    float alpha=1.f;
    float beta=0.f;
    std::cout << "m_num:" << m_num << endl <<  "m_feat_len" << m_feat_len<<endl;
	cblas_sgemv(CblasRowMajor, CblasNoTrans, m_num, m_feat_len, alpha, m_pfs, m_feat_len, m_fi, 1, beta, m_res, 1);
	
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
}

