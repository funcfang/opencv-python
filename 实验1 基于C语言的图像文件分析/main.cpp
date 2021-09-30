#include <stdio.h>
#include <malloc.h>
#include "BitmapFormat.h"

class BMP_reader
{
private:
	
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	
public:
	void read(char* fileName)
	{
		FILE* fp;
		fp = fopen(fileName, "rb");//读取同目录下的image.bmp文件。
		if (fp == NULL)
		{
			printf("打开'image.bmp'失败！\n");
			return ;
		}
		//如果不先读取bifType，根据C语言结构体Sizeof运算规则——整体大于部分之和，从而导致读文件错位
		unsigned short  fileType;
		fread(&fileType, 1, sizeof(unsigned short), fp);
		if (fileType = 0x4d42)
		{
			printf("文件类型标识正确!");
			printf("\n文件标识符：%d\n", fileType);
			fread(&fileHeader,sizeof(BITMAPFILEHEADER), 1,  fp);
			showBmpHead(fileHeader);
			fread(&infoHeader, sizeof(BITMAPINFOHEADER),1,  fp);
			showBmpInfoHead(infoHeader);
		}
		fclose(fp);
		return;
	}
	
	void showBmpHead(BITMAPFILEHEADER pBmpHead)
	{  //定义显示信息的函数，传入文件头结构体
		printf("BMP文件大小：%dkb\n", pBmpHead.bfSize / 1024);
		printf("保留字必须为0：%d\n", pBmpHead.bfReserved1);
		printf("保留字必须为0：%d\n", pBmpHead.bfReserved2);
		printf("实际位图数据的偏移字节数: %d\n", pBmpHead.bfOffBits);
	}
	
	void showBmpInfoHead(BITMAPINFOHEADER pBmpinfoHead)
	{//定义显示信息的函数，传入的是信息头结构体
		printf("位图信息头:\n");
		printf("信息头的大小:%d\n", pBmpinfoHead.biSize);
		printf("位图宽度:%d\n", pBmpinfoHead.biWidth);
		printf("位图高度:%d\n", pBmpinfoHead.biHeight);
		printf("图像的位面数(位面数是调色板的数量,默认为1个调色板):%d\n", pBmpinfoHead.biPlanes);
		printf("每个像素的位数:%d\n", pBmpinfoHead.biBitCount);
		printf("压缩方式:%d\n", pBmpinfoHead.biCompression);
		printf("图像的大小:%d\n", pBmpinfoHead.biSizeImage);
		printf("水平方向分辨率:%d\n", pBmpinfoHead.biXPelsPerMeter);
		printf("垂直方向分辨率:%d\n", pBmpinfoHead.biYPelsPerMeter);
		printf("使用的颜色数:%d\n", pBmpinfoHead.biClrUsed);
		printf("重要颜色数:%d\n", pBmpinfoHead.biClrImportant);
	}
	
	//修改图片并保存 
	void update_bmp(char* fileName,char* savedFileName)
	{
		printf("\n将读取的图片中间1/4面积像素改为0，并且保存图片。\n");
		
		FILE* fp;
		fp = fopen(fileName, "rb");
		if (fp == NULL)
		{
			printf("打开'image.bmp'失败！\n");
			return ;
		}

		unsigned short  fileType;
		fread(&fileType, 1, sizeof(unsigned short), fp);
		if (fileType = 0x4d42)
		{
			printf("文件类型标识正确!");
			printf("\n文件标识符：%d\n", fileType);
		
			//图像数据的操作
			fseek(fp, fileHeader.bfOffBits, SEEK_SET);
			unsigned char *r, *g, *b;
			r = (unsigned char *)malloc(sizeof(unsigned char)*infoHeader.biWidth*infoHeader.biHeight);
			b = (unsigned char *)malloc(sizeof(unsigned char)*infoHeader.biWidth*infoHeader.biHeight);
			g = (unsigned char *)malloc(sizeof(unsigned char)*infoHeader.biWidth*infoHeader.biHeight);
			int i, j;
			unsigned char pixVal = '\0';
			for (i = 0; i < infoHeader.biHeight; i++)
			{
				for (j = 0; j < infoHeader.biWidth; j++)
				{	
					fread(&pixVal, sizeof(unsigned char), 1, fp);
					*(r + infoHeader.biWidth * i + j) = pixVal;
					fread(&pixVal, sizeof(unsigned char), 1, fp);
					*(g + infoHeader.biWidth * i + j) = pixVal;
					fread(&pixVal, sizeof(unsigned char), 1, fp);
					*(b + infoHeader.biWidth * i + j) = pixVal;
				}
			}
			fclose(fp);
			
			//存储图像
			FILE* fpout;
			fpout = fopen(savedFileName, "wb");
			fwrite(&fileType, sizeof(unsigned short), 1, fpout);
			fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fpout);
			fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, fpout);	
			for (j = 0; j < infoHeader.biHeight; j++)
			{
				bool flag = false;  //默认不处于图片中间
				if( j>= (0.25)*infoHeader.biHeight && j<= (0.75)*infoHeader.biHeight) flag = true;
				for (i = 0; i < infoHeader.biWidth; i++)
				{
					if(flag&&( i >= (0.25)*infoHeader.biWidth ) && ( i <= (0.75)*infoHeader.biWidth ))
					{
						pixVal = '0';
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);	
					}else{
						pixVal = r[infoHeader.biWidth * j + i];
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
						pixVal = g[infoHeader.biWidth * j + i];
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
						pixVal = b[infoHeader.biWidth * j + i];
						fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
					}
					
				}
			}
			printf("修改成功，另存为 %s\n",savedFileName);
			fclose(fpout);
		}
	}
};

int main()
{
	BMP_reader reader;
    reader.read("demo.bmp");
    reader.update_bmp("demo.bmp","savedOK.bmp"); 
	return 0;
}
