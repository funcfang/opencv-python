
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
		fp = fopen(fileName, "rb");//��ȡͬĿ¼�µ�image.bmp�ļ���
		if (fp == NULL)
		{
			printf("��'image.bmp'ʧ�ܣ�\n");
			return ;
		}
		//������ȶ�ȡbifType������C���Խṹ��Sizeof������򡪡�������ڲ���֮�ͣ��Ӷ����¶��ļ���λ
		unsigned short  fileType;
		fread(&fileType, 1, sizeof(unsigned short), fp);
		if (fileType = 0x4d42)
		{
			printf("�ļ����ͱ�ʶ��ȷ!");
			printf("\n�ļ���ʶ����%d\n", fileType);
			fread(&fileHeader,sizeof(BITMAPFILEHEADER), 1,  fp);
			showBmpHead(fileHeader);
			fread(&infoHeader, sizeof(BITMAPINFOHEADER),1,  fp);
			showBmpInfoHead(infoHeader);
		}
		fclose(fp);
		return;
	}
	
	void showBmpHead(BITMAPFILEHEADER pBmpHead)
	{  //������ʾ��Ϣ�ĺ����������ļ�ͷ�ṹ��
		printf("BMP�ļ���С��%dkb\n", pBmpHead.bfSize / 1024);
		printf("�����ֱ���Ϊ0��%d\n", pBmpHead.bfReserved1);
		printf("�����ֱ���Ϊ0��%d\n", pBmpHead.bfReserved2);
		printf("ʵ��λͼ���ݵ�ƫ���ֽ���: %d\n", pBmpHead.bfOffBits);
	}
	
	void showBmpInfoHead(BITMAPINFOHEADER pBmpinfoHead)
	{//������ʾ��Ϣ�ĺ��������������Ϣͷ�ṹ��
		printf("λͼ��Ϣͷ:\n");
		printf("��Ϣͷ�Ĵ�С:%d\n", pBmpinfoHead.biSize);
		printf("λͼ���:%d\n", pBmpinfoHead.biWidth);
		printf("λͼ�߶�:%d\n", pBmpinfoHead.biHeight);
		printf("ͼ���λ����(λ�����ǵ�ɫ�������,Ĭ��Ϊ1����ɫ��):%d\n", pBmpinfoHead.biPlanes);
		printf("ÿ�����ص�λ��:%d\n", pBmpinfoHead.biBitCount);
		printf("ѹ����ʽ:%d\n", pBmpinfoHead.biCompression);
		printf("ͼ��Ĵ�С:%d\n", pBmpinfoHead.biSizeImage);
		printf("ˮƽ����ֱ���:%d\n", pBmpinfoHead.biXPelsPerMeter);
		printf("��ֱ����ֱ���:%d\n", pBmpinfoHead.biYPelsPerMeter);
		printf("ʹ�õ���ɫ��:%d\n", pBmpinfoHead.biClrUsed);
		printf("��Ҫ��ɫ��:%d\n", pBmpinfoHead.biClrImportant);
	}
	
	//�޸�ͼƬ������ 
	void update_bmp(char* fileName,char* savedFileName)
	{
		printf("\n����ȡ��ͼƬ�м�1/4������ظ�Ϊ0�����ұ���ͼƬ��\n");
		
		FILE* fp;
		fp = fopen(fileName, "rb");
		if (fp == NULL)
		{
			printf("��'image.bmp'ʧ�ܣ�\n");
			return ;
		}

		unsigned short  fileType;
		fread(&fileType, 1, sizeof(unsigned short), fp);
		if (fileType = 0x4d42)
		{
			printf("�ļ����ͱ�ʶ��ȷ!");
			printf("\n�ļ���ʶ����%d\n", fileType);
		
			//ͼ�����ݵĲ���
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
			
			//�洢ͼ��
			FILE* fpout;
			fpout = fopen(savedFileName, "wb");
			fwrite(&fileType, sizeof(unsigned short), 1, fpout);
			fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fpout);
			fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, fpout);	
			for (j = 0; j < infoHeader.biHeight; j++)
			{
				bool flag = false;  //Ĭ�ϲ�����ͼƬ�м�
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
			printf("�޸ĳɹ������Ϊ %s\n",savedFileName);
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
