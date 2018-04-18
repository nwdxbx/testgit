#include "PutChineseToMat.h"
#include "CvxText.h"

/*
	To test the library, include "PutChineseToMat.h" from an application project
	and call PutChineseToMatTest().
	
	Do not forget to add the library to Project Dependencies in Visual Studio.
*/

//char* sFont = "/usr/share/fonts/truetype/freefont/simhei.ttf";
//sFont = "/home/vis/vgdb/e/visystem/VisImageProcess/Src/visStructServer2.0/Debug/simhei.ttf";
//sFont = "/home/vis/vgdb/e/visystem/VisImageProcess/Src/visStructServer2.0/Debug/SIMSUNB.TTF";
//sFont = "/home/vis/vgdb/e/visystem/VisImageProcess/Src/visStructServer2.0/Debug/TAHOMABD.TTF";
//	sFont = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";
//	sFont = "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-C.ttf";

const char* gsFontPath = "/usr/share/fonts-font-awesome/fonts/fontawesome-webfont.ttf";

void SetPutText(const char* const sFontPath)
{
	gsFontPath = sFontPath;
}

// ������ֲ���:
//
// img			-ipl ͼ��
// text         -�����������Ϣ
// pos			-��ʾλ��
// color		-������ɫ
// size			-�����С
// interval     -������
//���磺PutText_2_IplImg(image, "�й�", cvPoint(20, 100), CV_RGB(128,255,127),30,0.2);
void PutText_2_IplImg(Mat& img, const char *text, Point pos, Scalar color, int size, double interval)
{
	CvxText Puttext(gsFontPath);//
	if (img.empty())
	{
		return;
	}
	if(interval >= 1.0)
		interval /= 10;
	// size         - �����С
	Scalar Scalar1=Scalar(size,0,interval,0);
	float p = 0.9f;
	Puttext.setFont(NULL, &Scalar1,true, &p);   // ͸������
	Puttext.putText(img, text, pos,color);
	//CATCH_OPENCV("text ::PutText_2_IplImg")
	return;
}

PutTextToMat_HANDLE PutText_Open(int ID /*= 0*/)
{
	return (PutTextToMat_HANDLE)(new CvxText(gsFontPath));
}


bool PutText_Close(PutTextToMat_HANDLE* hHandle)
{
	if(!hHandle)return false;
	if(!(*hHandle))return false;
	delete ((CvxText*)(*hHandle));
	(*hHandle) = NULL;
	return true;
}


bool PutText2Ipl(PutTextToMat_HANDLE hHandle, Mat& img, const char*text, Point pos, Scalar color, int size, double interval)
{
	if(!hHandle)return false;
	if(interval >= 1.0)
		interval /= 10;
	cv::Scalar Scalar1=cv::Scalar(size,0,interval,0);
	float p = 0.9f;
	((CvxText*)hHandle)->setFont(NULL, &Scalar1, NULL, &p);   // ͸������
	return ((CvxText*)hHandle)->putText(img, text, pos,color);
}
