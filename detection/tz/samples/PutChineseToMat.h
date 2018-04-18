#pragma once

#define PutTextToMat_HANDLE void*
#include <opencv2/core/core.hpp>
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif


	void SetPutText(const char* const sFontPath);
// 输出文字参数:
	//
	// img			-ipl 图像
	// text         -输出的文字信息
	// pos			-显示位置
	// color		-字体颜色
	// size			-字体大小
	// interval     -字体间距 单位像素
	//例如：PutText_2_IplImg(image, "中国", cvPoint(20, 100), CV_RGB(128,255,127),30,1);
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
