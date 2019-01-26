#ifndef __DATA_LOADER_H__
#define __DATA_LOADER_H__

#include <fstream>

using namespace std;

#define GRB4(a)  ((unsigned int)((((unsigned char*)(a))[0] << 24) | (((unsigned char*)(a))[1] << 16) |  \
		        (((unsigned char*)(a))[2] <<  8) | ((unsigned char*)(a))[3]))

#define DATA_INPUT_IMAGE_FLAG 0x00000803

#define DATA_INPUT_LABEL_FLAG 0x00000801

// 用以加载读取数据的类
class dataLoader
{
public:
	dataLoader();
	~dataLoader();
public:
	void reset();
	bool openLabelFile(const char* url);
	bool openImageFile(const char* url);

	bool readIndex(int* label, int pos);
	bool readImage(char imageBuf[], int pos);

	bool read(int* label, char imageBuf[], int pos);

	inline int numLable() { return mNumLabel; }
	inline int numImage() { return mNumImage; }

	inline int labelLength() { return mLabelLen; }
	inline int imageLength() { return mImageLen; }

	inline int imageWidth() { return mImageWidth; }
	inline int imageHeight() { return mImageHeight; }
private:
	int mNumLabel;
	int mNumImage;

	int mLabelLen;
	int mImageLen;
	int mImageWidth;
	int mImageHeight;
	int mImageStartPos;
	int mLableStartPos;
	fstream mLabelFile;
	fstream mImageFile;
};

#endif