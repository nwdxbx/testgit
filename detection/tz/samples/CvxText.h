//====================================================================
//====================================================================
//
// �ļ�: CvxText.h
//
// ˵��: OpenCV�������
//
// ʱ��:
//
// ����: ����

//
//====================================================================
//====================================================================

#ifndef OPENCV_CVX_TEXT_20171028_H
#define OPENCV_CVX_TEXT_20171028_H

/**
* \file CvxText.h
* \brief OpenCV��������ӿ�
*
* ʵ���˺���������ܡ�
*/


#include <opencv2/opencv.hpp>
using namespace cv;

#include <ft2build.h>

#include FT_FREETYPE_H
#include FT_ERRORS_H
#include FT_MODULE_ERRORS_H
#include FT_SYSTEM_H

#include FT_IMAGE_H
#include FT_GLYPH_H
#include FT_BITMAP_H

#define  DEBUG_TEXT 0


/**
* \class CvxText
* \brief OpenCV���������
*
* OpenCV��������֡��ֿ���ȡ�����˿�Դ��FreeFype�⡣����FreeFype��
* GPL��Ȩ�����Ŀ⣬��OpenCV��Ȩ����һ�£����Ŀǰ��û�кϲ���OpenCV
* ��չ���С�
*
* ��ʾ���ֵ�ʱ����Ҫһ�������ֿ��ļ����ֿ��ļ�ϵͳһ�㶼�Դ��ˡ�
* ������õ���һ����Դ���ֿ⣺����Ȫ�������塱��
*
* ����"OpenCV��չ��"��ϸ�������
* http://code.google.com/p/opencv-extension-library/
*
* ����FreeType��ϸ�������
* http://www.freetype.org/
*
*/

class CvxText 
{
   // ��ֹcopy
   CvxText& operator=(const CvxText&);
public:

   /**
    * װ���ֿ��ļ�
    */
   CvxText(const char *freeType);
   virtual ~CvxText();

   //================================================================
   //================================================================

   /**
    * ��ȡ���塣Ŀǰ��Щ�����в�֧�֡�
    *
    * \param font        ��������, Ŀǰ��֧��
    * \param size        �����С/�հױ���/�������/��ת�Ƕ�
    * \param underline   �»���
    * \param diaphaneity ͸����
    *
    * \sa setFont, restoreFont
    */

   void getFont(int *type,Scalar *size=NULL, bool *underline=NULL, float *diaphaneity=NULL);

   /**
    * �������塣Ŀǰ��Щ�����в�֧�֡�
    *
    * \param font        ��������, Ŀǰ��֧��
    * \param size        �����С/�հױ���/�������/��ת�Ƕ�
    * \param underline   �»���
    * \param diaphaneity ͸����
    *
    * \sa getFont, restoreFont
    */
   void setFont(int *type, Scalar *size=NULL, bool underline=false, float *diaphaneity=NULL);


   /**
    * ������֡���������������ַ���ֹͣ��
    *
    * \param img   �����Ӱ��
    * \param text  �ı�����
    * \param pos   �ı�λ��
    * \param color �ı���ɫ
    *
    * \return ���سɹ�������ַ����ȣ�ʧ�ܷ���-1��
    */
   int putText(Mat& img, const char * text, Point pos, Scalar color);

   //================================================================
   //================================================================

private:

   // �����ǰ�ַ�, ����m_posλ��

   void putWChar(Mat& img, wchar_t wc, Point &pos, Scalar color);

   //================================================================
   //================================================================

private:

   FT_Library   m_library;   // �ֿ�
   FT_Face      m_face;      // ����

   //================================================================
   //================================================================

   // Ĭ�ϵ������������

   int			m_fontType;
   Scalar		m_fontSize;
   bool			m_fontUnderline;
   float		m_fontDiaphaneity;
 

   //================================================================
   //================================================================
};

#endif // OPENCV_CVX_TEXT_2007_08_31_H

