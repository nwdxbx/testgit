#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>
#include <iconv.h>

#include "CvxText.h"


/*代码转换:从一种编码转为另一种编码*/  
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

/*GB2312码转为utf-8码*/  
int GB2312_To_UTF8(const char *inbuf,size_t inlen,unsigned char *outbuf,size_t outlen)  
{  
	return code_convert("gb2312","utf-8",inbuf,inlen,outbuf,outlen);  
}  

unsigned short Get_Unicode(char *in_gb2312)  
{  
	unsigned char out[256];  
	int rc;  
	unsigned int length_gb2312;  

	/* gb2312码转为utf8码 */  
	length_gb2312 = strlen(in_gb2312);  
	rc = GB2312_To_UTF8(in_gb2312,length_gb2312,out,256);  

	/* utf8转unicode码 */  
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

/* 功能：在屏幕上以0和1表示字体位图 */  
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
	char *font_file,             /* 字体文件路径 */  
	char *in_text,               /* 输入的字符串 */  
	int fonts_pixel_size,        /* 字体大小 */  
	int space,                   /* 间距 */  
	Font_Bitmap_Data *out_fonts  /* 输出的位图数据,Pic_Data是结构体 */  
	)  
	/* ****************************************** */  
	/* 根据传入的字体库路径、字符串、字体尺寸，输 */  
	/* 出该字符串的位图数组                       */  
	/* 本函数使用了freetype2的API                 */  
	/* ****************************************** */  
{  
	FT_Library         p_FT_Lib = NULL;    /* 库的句柄  */  
	FT_Face            p_FT_Face = NULL;      /* face对象的句柄 */  
	FT_Error           error = 0;  
	FT_Bitmap          bitmap;  
	FT_BitmapGlyph     bitmap_glyph;  
	FT_Glyph           glyph;  
	FT_GlyphSlot       slot;  
	int i , j ,temp,num,bg_height;  
	char error_str[200];  
	error = FT_Init_FreeType( & p_FT_Lib);  /* 初始化FreeType库 */  
	if (error)   /* 当初始化库时发生了一个错误 */  
	{  
		p_FT_Lib = 0 ;  
		//printf(FT_INIT_ERROR);  
		return - 1 ;  
	}  
	/* 从字体库文件中获取字体 */  
	error = FT_New_Face(p_FT_Lib, font_file , 0 , & p_FT_Face);  
	if ( error == FT_Err_Unknown_File_Format )   
	{   
		//printf(FT_UNKNOWN_FILE_FORMAT); /* 未知文件格式 */  
		return - 1 ;  
	}   
	else if (error)  
	{  
		//printf(FT_OPEN_FILE_ERROR);/* 打开错误 */  
		perror("FreeeType2");  
		return - 1 ;  
	}  
	j = 0;  
	wchar_t *unicode_text;/* 用于存储unicode字符 */  
	char ch[256];  
	unicode_text = (wchar_t*)calloc(1,sizeof(wchar_t)*(strlen(in_text)*2));/* 申请内存 */  
	for(i=0;i<strlen(in_text);++i)
	{  
		memset(ch,0,sizeof(ch));  /* 所有元素置零 */  
		ch[0] = in_text[i];  /* 获取in_text中的第i个元素 */  
		if(ch[0] < 0) 
		{   
			/* GB2312编码的汉字，每byte是负数，因此，可以用来判断是否有汉字 */  
			if(i < strlen(in_text)-1){/* 如果没有到字符串末尾 */  
				ch[1] = in_text[i+1];  
				++i;  
			}  
			else break;  
		}  
		unicode_text[j] = Get_Unicode(ch);  /* 开始转换编码 */  
		++j;  
	}  
	num = j; /* 记录字符的数量 */  

	int start_x = 0,start_y = 0;  
	int ch_height = 0,ch_width = 0;  
	int k,text_width = 0;  
	size_t size = 0;  
	unsigned char **text_alpha;   
	bg_height = fonts_pixel_size+5; /* 背景图形的高度，这个高度要大于字体的高度，所以是+5 */  
	/* 分配内存，用于存储字体背景图的数据 */  
	text_alpha = (unsigned char**)malloc(sizeof(unsigned char*)*bg_height);   
	for(i=0;i<bg_height;++i){  
		/* 预先为背景图的每一行分配内存 */  
		text_alpha[i] = (unsigned char*)malloc(sizeof(unsigned char)*1);   
	}  
	FT_Select_Charmap(p_FT_Face,FT_ENCODING_UNICODE);   /* 设定为UNICODE，默认的也是 */  
	FT_Set_Pixel_Sizes(p_FT_Face,0,fonts_pixel_size);   /* 设定字体大小 */  

	slot = p_FT_Face->glyph;  
	for(temp=0;temp<num;++temp){  
		/* 开始遍历unicode编码的字符串中的每个元素  */  
		/* 这个函数只是简单地调用FT_Get_Char_Index和FT_Load_Glyph */  
		error = FT_Load_Char( p_FT_Face, unicode_text[temp],  FT_LOAD_RENDER | FT_LOAD_NO_AUTOHINT);   
		if(!error){  
			/* 从插槽中提取一个字形图像 */  
			/* 请注意，创建的FT_Glyph对象必须与FT_Done_Glyph成对使用 */  
			error = FT_Get_Glyph(p_FT_Face -> glyph, &glyph);  
			if (!error)  
			{  
				if(unicode_text[temp] == ' ') {  
					/* 如果有空格 */  
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
					start_x += (ch_width+space); /* 画笔向右边移动 */  
				}  
				else{  
					/* 256级灰度字形转换成位图 */  
					FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, 0 ,1);  
					/* FT_RENDER_MODE_NORMAL       这是默认渲染模式，它对应于8位抗锯齿位图。 */  
					bitmap_glyph = (FT_BitmapGlyph)glyph;  
					bitmap       = bitmap_glyph -> bitmap;  
					k = 0;  

					start_y = fonts_pixel_size - slot->bitmap_top + 2; /* 获取起点的y轴坐标 */  
					if(start_y < 0) start_y = 0;  
					if(bitmap.rows > bg_height) ch_height = fonts_pixel_size;  
					else ch_height = bitmap.rows;  
					if(ch_height+start_y > bg_height) ch_height = bg_height - start_y;  
					ch_width = bitmap.width;  

					text_width = start_x + bitmap.width;  
					for(i=0;i<bg_height;++i){  
						/* 动态扩增存储字体位图的背景图形占用的空间 */  
						text_alpha[i] = (unsigned char*)realloc(text_alpha[i],sizeof(unsigned char)*text_width);  
						for(j=start_x-space;j<text_width;++j) text_alpha[i][j] = 0;/* 多出来的空间全部置零 */  
					}  
					/* 开始将字体位图贴到背景图形中 */  
					for(i = 0; i < bg_height; ++i){   
						for(j = 0;j < ch_width; ++j){  
							if(i >= start_y && i < start_y + ch_height){  
								/* 如果在字体位图的范围内 */  
								text_alpha[i][start_x + j] = bitmap.buffer[k];  
								++k;  
							}  
							else text_alpha[i][start_x + j] = 0;/* 否则就置零 */  
						}  
					}  
					start_x += (ch_width+space); /* 画笔向右边移动 */  
					/* 释放字形占用的内存 */  
					FT_Done_Glyph(glyph);  
					glyph = NULL;  
				}  
			}  
			else{  
				sprintf(error_str,"FreeType2 错误[%d]",error);  
				perror(error_str);  
			}  
		}  
		else{  
			sprintf(error_str,"FreeType2 错误[%d]",error);  
			perror(error_str);  
		}  
	}  
	/* 释放face占用的内存 */  
	FT_Done_Face(p_FT_Face);  
	p_FT_Face = NULL;  
	/* 释放FreeType Lib占用的内存 */  
	FT_Done_FreeType(p_FT_Lib);  
	p_FT_Lib = NULL;    
	temp = 0;  
	out_fonts->width    = text_width;          /* 要输出的位图的宽度 */  
	out_fonts->height   = bg_height;           /* 要输出的位图的高度 */  
	if(out_fonts->text_alpha) 
		free(out_fonts->text_alpha);  
	size = sizeof(unsigned char) * text_width * bg_height;  
	out_fonts->text_alpha = (unsigned char*)calloc(1,size);   /* 申请内存用来存储 */  
	k = 0;  
	for ( i = 0 ; i < bg_height; ++i)  
	{  
		for ( j = 0 ; j < text_width; ++j)  
		{  
			out_fonts->text_alpha[k] = text_alpha[i][j];  
			++k;  
		}  
	} 
	/* 释放内存 */  
	for(i=0;i<bg_height;++i){  
		free(text_alpha[i]);  
	}  
	free(text_alpha);  
	free(unicode_text);  
	return 0;  
} 
//====================================================================
//====================================================================


// 打开字库

CvxText::CvxText(const char *freeType)
{
	assert(freeType != NULL);

	// 打开字库文件, 创建一个字体

	if(FT_Init_FreeType(&m_library))
	{
		//MessageBox(NULL,"当前目录下没有'simhei.ttf'字体库！\n或者字体库不版本不正确！", "中文输出错误", MB_OK | MB_ICONEXCLAMATION);
		printf("error:FT_Init_FreeType no *ttf\n");
		exit(-2001);
	}
	if(FT_New_Face(m_library, freeType, 0, &m_face))
	{
		//MessageBox(NULL,"当前目录下没有'simhei.ttf'字体库！\n或者字体库不版本不正确！", "中文输出错误", MB_OK | MB_ICONEXCLAMATION);
		printf("error:FT_New_Face no *ttf\n");
		exit(-2002);
	}
	
	// 设置字体输出参数
	m_fontType = 0;            // 字体类型(不支持)

	m_fontSize.val[0] = 50;      // 字体大小
	m_fontSize.val[1] = 0.5;   // 空白字符大小比例
	m_fontSize.val[2] = 0.5;   // 间隔大小比例
	m_fontSize.val[3] = 0;      // 旋转角度(不支持)

	m_fontUnderline   = false;   // 下画线(不支持)

	m_fontDiaphaneity = 1.0;   // 色彩比例(可产生透明效果)

	// 设置字符大小

	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
	FT_Select_Charmap(m_face,FT_ENCODING_UNICODE);

	// 设置C语言的字符集环境
	//setlocale(LC_ALL, "");


// 	char* sFont = "/usr/share/fonts/truetype/freefont/simhei.ttf";
// 	sFont = "/home/vis/vgdb/e/visystem/VisImageProcess/Src/visStructServer2.0/Debug/simhei.ttf";
// 	Font_Bitmap_Data fontData;
// 	fontData.text_alpha = NULL;
// 	Get_Fonts_Bitmap(sFont,"中",20,2,&fontData);
// 	Show_Font_Bitmap(&fontData);
}

// 释放FreeType资源

CvxText::~CvxText()
{
	FT_Done_Face    (m_face);
	FT_Done_FreeType(m_library);
}

// 设置字体参数:
//
// font         - 字体类型, 目前不支持
// size         - 字体大小/空白比例/间隔比例/旋转角度
// underline   - 下画线
// diaphaneity   - 透明度

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
	// 参数合法性检查
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
		//FT_Set_Pixel_Sizes(m_face,0,(int)m_fontSize.val[0]);   /* 设定字体大小 */ 
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

		// 解析双字节符号
		if(!isascii(tmp_char[0]))
		{
			tmp_char[1] = text[i+1];
			++i;
			tmp_char[2]  = '\0';
		}
		unicode_text = Get_Unicode(tmp_char);

		//输出当前的字符
		putWChar(img, unicode_text, pos, color);
	}
	return 1;
}

// 输出当前字符, 更新m_pos位置
void CvxText::putWChar(Mat& img, wchar_t wc, Point &pos, Scalar color)
{
	// 根据unicode生成字体的二值位图
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

	/* 如果是空格 */ 
	//if(' ' != wc) 
	{
		FT_Vector origin; origin.x = 32; origin.y = 0;
		/* 256级灰度字形转换成位图*/  
		FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, &origin ,1);  
		/* FT_RENDER_MODE_NORMAL       这是默认渲染模式，它对应于8位抗锯齿位图。FT_RENDER_MODE_MONO */  
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
		/* 释放字形占用的内存 */  
		FT_Done_Glyph(glyph);  
	} 


	// 修改下一个字的输出位置

	double space = m_fontSize.val[0]*m_fontSize.val[1];
	double sep   = m_fontSize.val[0]*m_fontSize.val[2];

	pos.x += (int)((cols? cols: space) + sep);

}

