#pragma once

#define PutTextToMat_HANDLE void*
#include <opencv2/core/core.hpp>
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif


	void SetPutText(const char* const sFontPath);
// ������ֲ���:
	//
	// img			-ipl ͼ��
	// text         -�����������Ϣ
	// pos			-��ʾλ��
	// color		-������ɫ
	// size			-�����С
	// interval     -������ ��λ����
	//���磺PutText_2_IplImg(image, "�й�", cvPoint(20, 100), CV_RGB(128,255,127),30,1);
	void PutText_2_IplImg(Mat& img, const char *text, Point pos, Scalar color, int size, double interval);
	typedef void(*PPutText_2_IplImg)(Mat& img, const char *text, Point pos, Scalar color, int size, double interval);

	PutTextToMat_HANDLE  PutText_Open(int ID = 0);
	typedef PutTextToMat_HANDLE(*PPutText_Open)(int ID);

	bool PutText_Close(PutTextToMat_HANDLE* hHandle);
	typedef bool(*PPutText_Close)(PutTextToMat_HANDLE* hHandle);

	bool PutText2Ipl(PutTextToMat_HANDLE hHandle, Mat& img, const char*text, Point pos, Scalar color, int size, double interval);
	typedef bool(*PPutText2Ipl)(PutTextToMat_HANDLE hHandle, Mat& img, const char*text, Point pos, Scalar color, int size, double interval);

	
	
#ifdef __cplusplus
}
#endif
