/********************************************************************************* 
  *Copyright(C)
  *FileName: Compare.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2018-1-4
  *Description:  compare feature base functions
  *History:
   Date: 2018-1-4
   Author: Jin
   Modification: YES
**********************************************************************************/
#include "../../include/common.h"
#include "./Compare.h"
#include <malloc.h>

Compare::Compare(int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType)
{
	m_fi = NULL;
	m_res = NULL;
	m_pfs = NULL;
	m_pos=0;
	
	CHECK(feaitems > 0);
	m_num=feaitems;
	uuid_.resize(m_num, "EMPTY_ENTRY");
	isempty_.resize(m_num, true);

	switch (type){
	  case EM_FEATURE_CAL_PERSON:{
          m_feat_len = 128;
		  break;
	  }
	  case EM_FEATURE_CAL_FACE:{
          m_feat_len = 128;
		  break;
	  }
	}

    if (cmpType == EM_FEATURE_CMP_GPU)
    {
        m_fi = (float*)malloc(m_feat_len*sizeof(float));
        if (NULL == m_fi)
        {
            std::cout<< "[ERROR]: CompareCpu fi malloc error!" <<std::endl;
        }

        m_res = (float*)malloc(m_num*sizeof(float));
        if (NULL == m_res)
        {
            std::cout<< "[ERROR]: CompareCpu res malloc error!" <<std::endl;
        }

        memset(m_fi, 0, m_feat_len*sizeof(float));
        memset(m_res, 0, m_num*sizeof(float));
    }
    else
    {
        m_fi = (float*)malloc(m_feat_len*sizeof(float));
        if (NULL == m_fi)
        {
            std::cout<< "[ERROR]: CompareCpu fi malloc error!" <<std::endl;
        }

        if (cmpType != EM_FEATURE_CMP_MULTI_CPU)
        {
            m_res = (float*)malloc(m_num*sizeof(float));
            if (NULL == m_res)
            {
                std::cout<< "[ERROR]: CompareCpu res malloc error!" <<std::endl;
            }
        }

        m_pfs = (float*)malloc(m_num*m_feat_len*sizeof(float));
        if (NULL == m_pfs)
        {
            std::cout<< "[ERROR]: CompareCpu fs_ malloc error!" <<std::endl;
        }
        memset(m_fi, 0, m_feat_len*sizeof(float));
        memset(m_pfs, 0, m_num*m_feat_len*sizeof(float));
    }
}

Compare::~Compare()
{
	if (m_fi != NULL)
	{
		free(m_fi);
	}

	if (m_res != NULL)
	{
		free(m_res);
	}
	
	if (m_pfs != NULL)
	{
		free(m_pfs);
	}
}


void Compare::Parse(std::string &feat, std::vector<float> &res)
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

