/********************************************************************************* 
  *Copyright(C)
  *FileName: CompareApi.cpp
  *Author: Jin
  *Version: V1.0
  *Date: 2018-1-4
  *Description:  compare feature apis
  *History:
   Date: 2018-1-4
   Author: Jin
   Modification: YES
**********************************************************************************/

#include "../../include/tuzhenfeature.h"
#include "./Compare.h"
#include "./CompareGpu.h"
#include "./CompareCpu.h"
#include "./CompareMultiCpu.h"


int feacompare_open(void **ppvFeaCmpHandle, int deviceID, int feaitems, EFeature_Cal_Type type, EFeature_Cmp_Type cmpType)
{
	Compare *pCmp = NULL;
	void *pFeatCmpHandle = NULL;
	
	if (cmpType == EM_FEATURE_CMP_GPU)
	{
        pCmp = new CompareGpu(deviceID, feaitems, type, cmpType);
		pFeatCmpHandle = (void*)dynamic_cast<CompareGpu*>(pCmp);
		std::cout<< "Compare Type: GPU" << std::endl;
	}
	else if (cmpType == EM_FEATURE_CMP_CPU)
	{
        pCmp = new CompareCpu(feaitems, type, cmpType);
		pFeatCmpHandle = (void*)dynamic_cast<CompareCpu*>(pCmp);
		std::cout<< "Compare Type: CPU" << std::endl;
	}
    else if (cmpType == EM_FEATURE_CMP_MULTI_CPU)
    {
        pCmp = new CompareMultiCpu(feaitems, type, cmpType);
        pFeatCmpHandle = (void*)dynamic_cast<CompareMultiCpu*>(pCmp);
        std::cout<< "Compare Type: Multi CPU" << std::endl;
    }
	else
	{
		std::cout<< "[ERROR]: type[" << cmpType << "] is invalid!!" << std::endl;
		return EM_ERROR_NULL_POINT;
	}

    if(pFeatCmpHandle == NULL)
    {
    	std::cout<< "[ERROR]: pFeatCmpHandle is NULL!!" << std::endl;
        return EM_ERROR_MALLOC;
    }

    *ppvFeaCmpHandle = pFeatCmpHandle;
    return 0;
}


int feacompare_process(void *pvFeaCmpHandle, std::string &feature, std::vector<float> &res, std::vector<std::string> &uuid,float thresh)
{
    Compare *pCompare = (Compare*)pvFeaCmpHandle;
    pCompare->Process(feature, res, uuid,thresh);

    return 0;
}

int feacompare_addfeat(void *pvFeaCmpHandle, std::vector<std::string> &fea, std::vector<std::string> &uuid)
{
    Compare *pCompare = (Compare*)pvFeaCmpHandle;
    pCompare->Addfeat(fea, uuid);
    return 0;
}


void feacompare_close(void *pvFeaCmpHandle)
{
    Compare *pCompare = (Compare*)pvFeaCmpHandle;
    delete pCompare;
    return;
}


