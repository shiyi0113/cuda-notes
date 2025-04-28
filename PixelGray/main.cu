#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
// 灰度图处理
__global__ 
void colortoGrayscaleConvertion(unsigned char* pout, unsigned char* pin, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < width && row < height) {
		int grayOffset = row * width + col;
		int rgbOffset = grayOffset * 3;
		unsigned char r = pin[rgbOffset];
		unsigned char g = pin[rgbOffset + 1];
		unsigned char b = pin[rgbOffset + 2];
		float gray = 0.21f * r + 0.72f * g + 0.07f * b;
		pout[grayOffset] = static_cast<unsigned char>(gray);
	}
}
//模糊图处理
__global__
void blurKernel(unsigned char* pout, unsigned char* pin, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int BLUR_SIZE = 1;
	if (col < width && row < height) {
		int pixval = 0;  
		int pixels = 0;  //像素数
		for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {   // 遍历累加周围像素
			for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
				int curCol = col + blurCol;
				int curRow = row + blurRow;
				if (curCol >= 0 && curCol < width && curRow >= 0 && curRow < height) {
					pixval += pin[curRow * width + curCol];
					pixels++;
				}
			}
		}
		pout[row * width + col] = static_cast<unsigned char>(pixval / pixels);  //平均值
	}
}
void colortoGrayscale(unsigned char* pout_h, unsigned char* pin_h, int width, int height) {
	size_t size_in = width * height * 3 * sizeof(unsigned char);
	size_t size_out = width * height * sizeof(unsigned char);
	unsigned char* pin_d,* pout_d;
	cudaMalloc((void**)&pin_d, size_in);
	cudaMalloc((void**)&pout_d, size_out);
	cudaMemcpy(pin_d, pin_h, size_in, cudaMemcpyHostToDevice);
	colortoGrayscaleConvertion << <dim3(ceil(width / 16.0), ceil(height / 16.0), 1), dim3(16, 16, 1) >> > (pout_d, pin_d, width, height);
	cudaMemcpy(pout_h, pout_d, size_out, cudaMemcpyDeviceToHost);
	cudaFree(pin_d);
	cudaFree(pout_d);
}
int main() {
	int width = 76;
	int height = 62;
	size_t size_in = width * height * 3 * sizeof(unsigned char);
	size_t size_out = width * height * sizeof(unsigned char);
	unsigned char* pin_h = (unsigned char*)malloc(size_in);
	unsigned char* pout_h = (unsigned char*)malloc(size_out);
	// 随机生成RGB矩阵
	srand(time(0));
	for (int i = 0; i < width * height * 3; i++) {
		pin_h[i] = static_cast<unsigned char>(rand() % 256);
	}
	colortoGrayscale(pout_h, pin_h, width, height);
	// 打印对比结果
	for (int i = 0; i < width * height; i++) {
		int grayOffset = i;
		int rgbOffset = i * 3;
		unsigned char r = pin_h[rgbOffset];
		unsigned char g = pin_h[rgbOffset + 1];
		unsigned char b = pin_h[rgbOffset + 2];
		unsigned char gray = pout_h[grayOffset];
		std::cout << "RGB: (" << (int)r << ", " << (int)g << ", " << (int)b << ") -> Grayscale: " << (int)gray << std::endl;
	}
	free(pin_h);
	free(pout_h);
}