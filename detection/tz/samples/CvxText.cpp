#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>
#include <iconv.h>

#include "CvxText.h"


/*����ת��:��һ�ֱ���תΪ��һ�ֱ���*/  
int code_convert(char *from_charset,char *to_charset,const char *inbuf,size_t inlen,  
	unsigned char *outbuf,size_t outlen)  
{  
	iconv_t cd;  
	const char **pin = &inbuf;  
	unsigned char **pout = &outbuf;  
	cd = iconv_open(to_charset,from_charset);  
	if (cd==0) return -1;  
	memset(outbuf,0,outlen);  
	if (iconv(cd,(char**)pin,&inlen,(char**)pout,&outlen)==-1) return -1;  
	iconv_close(cd);  
	return 0;  
}  

/*GB2312��תΪutf-8��*/  
int GB2312_To_UTF8(const char *inbuf,size_t inlen,unsigned char *outbuf,size_t outlen)  
{  
	return code_convert("gb2312","utf-8",inbuf,inlen,outbuf,outlen);  
}  

unsigned short Get_Unicode(char *in_gb2312)  
{  
	unsigned char out[256];  
	int rc;  
	unsigned int length_gb2312;  

	/* gb2312��תΪutf8�� */  
	length_gb2312 = strlen(in_gb2312);  
	rc = GB2312_To_UTF8(in_gb2312,length_gb2312,out,256);  

	/* utf8תunicode�� */  
	unsigned short unicode;  
	unicode = out[0];  
	if (unicode >= 0xF0) {  
		unicode = (unsigned short) (out[0] & 0x07) << 18;  
		unicode |= (unsigned short) (out[1] & 0x3F) << 12;  
		unicode |= (unsigned short) (out[2] & 0x3F) << 6;  
		unicode |= (unsigned short) (out[3] & 0x3F);  
	} else if (unicode >= 0xE0) {  
		unicode = (unsigned short) (out[0] & 0x0F) << 12;  
		unicode |= (unsigned short) (out[1] & 0x3F) << 6;  
		unicode |= (unsigned short) (out[2] & 0x3F);  
	} else if (unicode >= 0xC0) {  
		unicode = (unsigned short) (out[0] & 0x1F) << 6;  
		unicode |= (unsigned short) (out[1] & 0x3F);  
	}  
	return unicode;  
}  

typedef struct _Font_Bitmap_Data
{
	int height;
	int width;
	unsigned char* text_alpha;
}Font_Bitmap_Data;

/* ���ܣ�����Ļ����0��1��ʾ����λͼ */  
int Show_Font_Bitmap(Font_Bitmap_Data *in_fonts)  
{  
	int x,y;  
	for(y=0;y<in_fonts->height;++y){  
		for(x=0;x<in_fonts->width;++x){  
			if(in_fonts->text_alpha[y*in_fonts->width+x]>0)  
				printf("1");  
			else printf("0");  
		}  
		printf("\n");  
	}  
	printf("\n");  
	return 0;  
}  

int Get_Fonts_Bitmap(  
	char *font_file,             /* �����ļ�·�� */  
	char *in_text,               /* ������ַ��� */  
	int fonts_pixel_size,        /* �����С */  
	int space,                   /* ��� */  
	Font_Bitmap_Data *out_fonts  /* �����λͼ����,Pic_Data�ǽṹ�� */  
	)  
	/* ****************************************** */  
	/* ���ݴ���������·�����ַ���������ߴ磬�� */  
	/* �����ַ�����λͼ����                       */  
	/* ������ʹ����freetype2��API                 */  
	/* ****************************************** */  
{  
	FT_Library         p_FT_Lib = NULL;    /* ��ľ��  */  
	FT_Face            p_FT_Face = NULL;      /* face����ľ�� */  
	FT_Error           error = 0;  
	FT_Bitmap          bitmap;  
	FT_BitmapGlyph     bitmap_glyph;  
	FT_Glyph           glyph;  
	FT_GlyphSlot       slot;  
	int i , j ,temp,num,bg_height;  
	char error_str[200];  
	error = FT_Init_FreeType( & p_FT_Lib);  /* ��ʼ��FreeType�� */  
	if (error)   /* ����ʼ����ʱ������һ������ */  
	{  
		p_FT_Lib = 0 ;  
		//printf(FT_INIT_ERROR);  
		return - 1 ;  
	}  
	/* ��������ļ��л�ȡ���� */  
	error = FT_New_Face(p_FT_Lib, font_file , 0 , & p_FT_Face);  
	if ( error == FT_Err_Unknown_File_Format )   
	{   
		//printf(FT_UNKNOWN_FILE_FORMAT); /* δ֪�ļ���ʽ */  
		return - 1 ;  
	}   
	else if (error)  
	{  
		//printf(FT_OPEN_FILE_ERROR);/* �򿪴��� */  
		perror("FreeeType2");  
		return - 1 ;  
	}  
	j = 0;  
	wchar_t *unicode_text;/* ���ڴ洢unicode�ַ� */  
	char ch[256];  
	unicode_text = (wchar_t*)calloc(1,sizeof(wchar_t)*(strlen(in_text)*2));/* �����ڴ� */  
	for(i=0;i<strlen(in_text);++i)
	{  
		memset(ch,0,sizeof(ch));  /* ����Ԫ������ */  
		ch[0] = in_text[i];  /* ��ȡin_text�еĵ�i��Ԫ�� */  
		if(ch[0] < 0) 
		{   
			/* GB2312����ĺ��֣�ÿbyte�Ǹ�������ˣ����������ж��Ƿ��к��� */  
			if(i < strlen(in_text)-1){/* ���û�е��ַ���ĩβ */  
				ch[1] = in_text[i+1];  
				++i;  
			}  
			else break;  
		}  
		unicode_text[j] = Get_Unicode(ch);  /* ��ʼת������ */  
		++j;  
	}  
	num = j; /* ��¼�ַ������� */  

	int start_x = 0,start_y = 0;  
	int ch_height = 0,ch_width = 0;  
	int k,text_width = 0;  
	size_t size = 0;  
	unsigned char **text_alpha;   
	bg_height = fonts_pixel_size+5; /* ����ͼ�εĸ߶ȣ�����߶�Ҫ��������ĸ߶ȣ�������+5 */  
	/* �����ڴ棬���ڴ洢���屳��ͼ������ */  
	text_alpha = (unsigned char**)malloc(sizeof(unsigned char*)*bg_height);   
	for(i=0;i<bg_height;++i){  
		/* Ԥ��Ϊ����ͼ��ÿһ�з����ڴ� */  
		text_alpha[i] = (unsigned char*)malloc(sizeof(unsigned char)*1);   
	}  
	FT_Select_Charmap(p_FT_Face,FT_ENCODING_UNICODE);   /* �趨ΪUNICODE��Ĭ�ϵ�Ҳ�� */  
	FT_Set_Pixel_Sizes(p_FT_Face,0,fonts_pixel_size);   /* �趨�����С */  

	slot = p_FT_Face->glyph;  
	for(temp=0;temp<num;++temp){  
		/* ��ʼ����unicode������ַ����е�ÿ��Ԫ��  */  
		/* �������ֻ�Ǽ򵥵ص���FT_Get_Char_Index��FT_Load_Glyph */  
		error = FT_Load_Char( p_FT_Face, unicode_text[temp],  FT_LOAD_RENDER | FT_LOAD_NO_AUTOHINT);   
		if(!error){  
			/* �Ӳ������ȡһ������ͼ�� */  
			/* ��ע�⣬������FT_Glyph���������FT_Done_Glyph�ɶ�ʹ�� */  
			error = FT_Get_Glyph(p_FT_Face -> glyph, &glyph);  
			if (!error)  
			{  
				if(unicode_text[temp] == ' ') {  
					/* ����пո� */  
					k = 0;  
					ch_width   = (fonts_pixel_size -2)/2;  
					ch_height  = fonts_pixel_size;  
					text_width = start_x + ch_width;  
					start_y = 0;  
					for(i=0;i<bg_height;++i){  
						text_alpha[i] = (unsigned char*)realloc(text_alpha[i],sizeof(unsigned char)*text_width);  
						for(j=start_x-space;j<text_width;++j) text_alpha[i][j] = 0;  
					}  
					for ( i = 0 ; i < ch_height; ++i)  
					{  
						for ( j = 0 ; j < ch_width; ++j)  
						{  
							text_alpha[start_y + i][start_x + j] = 0;  
							++k;  
						}  
					}  
					start_x += (ch_width+space); /* �������ұ��ƶ� */  
				}  
				else{  
					/* 256���Ҷ�����ת����λͼ */  
					FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, 0 ,1);  
					/* FT_RENDER_MODE_NORMAL       ����Ĭ����Ⱦģʽ������Ӧ��8λ�����λͼ�� */  
					bitmap_glyph = (FT_BitmapGlyph)glyph;  
					bitmap       = bitmap_glyph -> bitmap;  
					k = 0;  

					start_y = fonts_pixel_size - slot->bitmap_top + 2; /* ��ȡ����y������ */  
					if(start_y < 0) start_y = 0;  
					if(bitmap.rows > bg_height) ch_height = fonts_pixel_size;  
					else ch_height = bitmap.rows;  
					if(ch_height+start_y > bg_height) ch_height = bg_height - start_y;  
					ch_width = bitmap.width;  

					text_width = start_x + bitmap.width;  
					for(i=0;i<bg_height;++i){  
						/* ��̬�����洢����λͼ�ı���ͼ��ռ�õĿռ� */  
						text_alpha[i] = (unsigned char*)realloc(text_alpha[i],sizeof(unsigned char)*text_width);  
						for(j=start_x-space;j<text_width;++j) text_alpha[i][j] = 0;/* ������Ŀռ�ȫ������ */  
					}  
					/* ��ʼ������λͼ��������ͼ���� */  
					for(i = 0; i < bg_height; ++i){   
						for(j = 0;j < ch_width; ++j){  
							if(i >= start_y && i < start_y + ch_height){  
								/* ���������λͼ�ķ�Χ�� */  
								text_alpha[i][start_x + j] = bitmap.buffer[k];  
								++k;  
							}  
							else text_alpha[i][start_x + j] = 0;/* ��������� */  
						}  
					}  
					start_x += (ch_width+space); /* �������ұ��ƶ� */  
					/* �ͷ�����ռ�õ��ڴ� */  
					FT_Done_Glyph(glyph);  
					glyph = NULL;  
				}  
			}  
			else{  
				sprintf(error_str,"FreeType2 ����[%d]",error);  
				perror(error_str);  
			}  
		}  
		else{  
			sprintf(error_str,"FreeType2 ����[%d]",error);  
			perror(error_str);  
		}  
	}  
	/* �ͷ�faceռ�õ��ڴ� */  
	FT_Done_Face(p_FT_Face);  
	p_FT_Face = NULL;  
	/* �ͷ�FreeType Libռ�õ��ڴ� */  
	FT_Done_FreeType(p_FT_Lib);  
	p_FT_Lib = NULL;    
	temp = 0;  
	out_fonts->width    = text_width;          /* Ҫ�����λͼ�Ŀ�� */  
	out_fonts->height   = bg_height;           /* Ҫ�����λͼ�ĸ߶� */  
	if(out_fonts->text_alpha) 
		free(out_fonts->text_alpha);  
	size = sizeof(unsigned char) * text_width * bg_height;  
	out_fonts->text_alpha = (unsigned char*)calloc(1,size);   /* �����ڴ������洢 */  
	k = 0;  
	for ( i = 0 ; i < bg_height; ++i)  
	{  
		for ( j = 0 ; j < text_width; ++j)  
		{  
			out_fonts->text_alpha[k] = text_alpha[i][j];  
			++k;  
		}  
	} 
	/* �ͷ��ڴ� */  
	for(i=0;i<bg_height;++i){  
		free(text_alpha[i]);  
	}  
	free(text_alpha);  
	free(unicode_text);  
	return 0;  
} 
//====================================================================
//====================================================================


// ���ֿ�

CvxText::CvxText(const char *freeType)
{
	assert(freeType != NULL);

	// ���ֿ��ļ�, ����һ������

	if(FT_Init_FreeType(&m_library))
	{
		//MessageBox(NULL,"��ǰĿ¼��û��'simhei.ttf'����⣡\n��������ⲻ�汾����ȷ��", "�����������", MB_OK | MB_ICONEXCLAMATION);
		printf("error:FT_Init_FreeType no *ttf\n");
		exit(-2001);
	}
	if(FT_New_Face(m_library, freeType, 0, &m_face))
	{
		//MessageBox(NULL,"��ǰĿ¼��û��'simhei.ttf'����⣡\n��������ⲻ�汾����ȷ��", "�����������", MB_OK | MB_ICONEXCLAMATION);
		printf("error:FT_New_Face no *ttf\n");
		exit(-2002);
	}
	
	// ���������������
	m_fontType = 0;            // ��������(��֧��)

	m_fontSize.val[0] = 50;      // �����С
	m_fontSize.val[1] = 0.5;   // �հ��ַ���С����
	m_fontSize.val[2] = 0.5;   // �����С����
	m_fontSize.val[3] = 0;      // ��ת�Ƕ�(��֧��)

	m_fontUnderline   = false;   // �»���(��֧��)

	m_fontDiaphaneity = 1.0;   // ɫ�ʱ���(�ɲ���͸��Ч��)

	// �����ַ���С

	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
	FT_Select_Charmap(m_face,FT_ENCODING_UNICODE);

	// ����C���Ե��ַ�������
	//setlocale(LC_ALL, "");


// 	char* sFont = "/usr/share/fonts/truetype/freefont/simhei.ttf";
// 	sFont = "/home/vis/vgdb/e/visystem/VisImageProcess/Src/visStructServer2.0/Debug/simhei.ttf";
// 	Font_Bitmap_Data fontData;
// 	fontData.text_alpha = NULL;
// 	Get_Fonts_Bitmap(sFont,"��",20,2,&fontData);
// 	Show_Font_Bitmap(&fontData);
}

// �ͷ�FreeType��Դ

CvxText::~CvxText()
{
	FT_Done_Face    (m_face);
	FT_Done_FreeType(m_library);
}

// �����������:
//
// font         - ��������, Ŀǰ��֧��
// size         - �����С/�հױ���/�������/��ת�Ƕ�
// underline   - �»���
// diaphaneity   - ͸����

void CvxText::getFont(int *type, Scalar *size, bool*underline, float *diaphaneity)
{
	if(type) *type = m_fontType;
	if(size) *size = m_fontSize;
	if(underline)
		*underline = m_fontUnderline;
	if(diaphaneity) *diaphaneity = m_fontDiaphaneity;
}

void CvxText::setFont(int *type, Scalar *size, bool underline, float *diaphaneity)
{
	// �����Ϸ��Լ��
	if(type)
	{
		if(type >= 0) m_fontType = *type;
	}
	if(size)
	{
		m_fontSize.val[0] = fabs(size->val[0]);
		m_fontSize.val[1] = fabs(size->val[1]);
		m_fontSize.val[2] = fabs(size->val[2]);
		m_fontSize.val[3] = fabs(size->val[3]);
		FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
		//FT_Set_Pixel_Sizes(m_face,0,(int)m_fontSize.val[0]);   /* �趨�����С */ 
		//FT_Set_Pixel_Sizes(m_face, 16, 16);
		//FT_Set_Char_Size(m_face,16 << 6 ,  16 << 6 ,  (int)m_fontSize.val[0] ,  (int)m_fontSize.val[0] );
	}
	m_fontUnderline   = underline;
	
	if(diaphaneity)
	{
		m_fontDiaphaneity = *diaphaneity;
	}
}

int CvxText::putText(Mat&img, const char* text, Point pos, Scalar color)
{
	if(img.empty()) return 0;
	if(text == NULL) return 0;

	//
	int i = 0;
	char tmp_char[4] = "";
	wchar_t unicode_text=0;
	for(i = 0; text[i] != '\0'; ++i)
	{

		tmp_char[0] = text[i];
		tmp_char[1] = '\0';

		// ����˫�ֽڷ���
		if(!isascii(tmp_char[0]))
		{
			tmp_char[1] = text[i+1];
			++i;
			tmp_char[2]  = '\0';
		}
		unicode_text = Get_Unicode(tmp_char);

		//�����ǰ���ַ�
		putWChar(img, unicode_text, pos, color);
	}
	return 1;
}

// �����ǰ�ַ�, ����m_posλ��
void CvxText::putWChar(Mat& img, wchar_t wc, Point &pos, Scalar color)
{
	// ����unicode��������Ķ�ֵλͼ
	FT_Error           error = 0;  
	FT_Bitmap          bitmap;  
	FT_BitmapGlyph     bitmap_glyph;  
	FT_Glyph           glyph;  
	FT_GlyphSlot       slot;
	char error_str[200] = "";
	error = FT_Load_Char(m_face,wc,FT_LOAD_RENDER|FT_LOAD_NO_AUTOHINT);   
	if(error)
	{
		sprintf(error_str,"FreeType2  FT_Load_Char errorcode:[%d]",error);  
		perror(error_str);
		return;
	}

	error = FT_Get_Glyph(m_face->glyph, &glyph);
	if(error)
	{
		sprintf(error_str,"FreeType2  FT_Get_Glyph errorcode:[%d]",error);  
		perror(error_str);
		return;
	}

	int rows = 0;
	int cols = 0;

	/* ����ǿո� */ 
	//if(' ' != wc) 
	{
		FT_Vector origin; origin.x = 32; origin.y = 0;
		/* 256���Ҷ�����ת����λͼ*/  
		FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, &origin ,1);  
		/* FT_RENDER_MODE_NORMAL       ����Ĭ����Ⱦģʽ������Ӧ��8λ�����λͼ��FT_RENDER_MODE_MONO */  
		bitmap_glyph = (FT_BitmapGlyph)glyph;  
		bitmap       = bitmap_glyph->bitmap; 

		rows = bitmap.rows;
		cols = bitmap.width;

		int width = 0;
		int height = 0;
		int ch = 3;
		if (/*img->roi*/0)
		{
			//w = img->roi->width;
			//h = img->roi->height;
		}else{
			width = img.cols;
			height = img.rows;
			ch = img.channels();
		}
		for(int h = 0; h < rows; ++h)
		{
			int tmpH = h + pos.y;
			if(tmpH >= height)
				break;
			uchar* pData = img.data + tmpH*width*ch;

			for(int w = 0; w < cols; ++w)
			{
				int offSet  = offSet = cols*h + w;
				int tmpw = w + pos.x;
				if(tmpw >= width)
					break;

				if(bitmap.buffer[offSet] > 1)
				{
					for (int iCh=0; iCh<ch; iCh++)
					{
						pData[tmpw*ch+iCh] = color.val[iCh];/*pData[w*ch+iCh]*(1-m_fontDiaphaneity) + color.val[k]*m_fontDiaphaneity;*/
					}
				}
			}
		}

		//pos.x += m_fontSize.val[0]*m_fontSize.val[2];
		//pos.x += cols;
		/* �ͷ�����ռ�õ��ڴ� */  
		FT_Done_Glyph(glyph);  
	} 


	// �޸���һ���ֵ����λ��

	double space = m_fontSize.val[0]*m_fontSize.val[1];
	double sep   = m_fontSize.val[0]*m_fontSize.val[2];

	pos.x += (int)((cols? cols: space) + sep);

}

