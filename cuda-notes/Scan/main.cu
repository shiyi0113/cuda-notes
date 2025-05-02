#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>

#define VECTOR_SIZE 2048

void sequential_scan(float* x_h, float* y_h, int n) {
	y_h[0] = x_h[0];
	for (size_t i = 1; i < n; i++) {
		y_h[i] = x_h[i] + y_h[i - 1];
	}
}
bool areFloatArrayEqual(const float* x, const float* y, int size, float epsilon = 1e-1) {
	for (size_t i = 0; i < size; i++) {
		if (std::fabs(x[i] - y[i]) > epsilon) {
			std::cout << "Mismatch at index " << i << ":" << x[i] << " vs " << y[i] << std::endl;
			return false;
		}
	}
	return true;
}

__global__
void kogge_stone_scan_kernel(float* x_d, float* y_d, int n) {
	extern __shared__ float XY[];
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		XY[threadIdx.x] = x_d[i];
	}
	else {
		XY[threadIdx.x] = 0.0f;
	}
	for (size_t step = 1; step < blockDim.x; step *= 2) {
		__syncthreads();
		float temp = 0.0f;
		if (threadIdx.x >= step) {
			temp = XY[threadIdx.x] + XY[threadIdx.x - step];
		}
		__syncthreads();
		if (threadIdx.x >= step) {
			XY[threadIdx.x] = temp;
		}
	}
	if (i < n) {
		y_d[i] = XY[threadIdx.x];
	}
}
void kogge_stone_scan(float* x_h, float* y_h, int n) {
	float* x_d, * y_d;
	cudaMalloc((void**)&x_d, n * sizeof(float));
	cudaMalloc((void**)&y_d, n * sizeof(float));
	cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(n);
	dim3 blockNum((n + blockSize.x - 1) / blockSize.x);
	kogge_stone_scan_kernel << <blockNum, blockSize, blockSize.x * sizeof(float) >> > (x_d, y_d, n);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
}

__global__
void brent_kung_scan_kernel(float* x_d, float* y_d, int n) {
	extern __shared__ float XY[];
	size_t i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) 
		XY[threadIdx.x] = x_d[i];
	else
		XY[threadIdx.x] = 0.0f;
	if (i + blockDim.x < n)
		XY[threadIdx.x + blockDim.x] = x_d[i + blockDim.x];
	else
		XY[threadIdx.x + blockDim.x] = 0.0f;
	for (size_t step = 1; step <= blockDim.x; step *= 2) {
		__syncthreads();
		size_t index = (threadIdx.x + 1) * 2 * step - 1;
		if (index < n)
			XY[index] += XY[index - step];
	}
	
	for (size_t step = blockDim.x / 2; step > 0; step /= 2) {
		__syncthreads();
		size_t index = (threadIdx.x + 1) * 2 * step - 1;
		if (index + step < n)
			XY[index + step] += XY[index];
	}
	__syncthreads();
	if (i < n)
		y_d[i] = XY[threadIdx.x];
	if (i + blockDim.x < n)
		y_d[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}
void brent_kung_scan(float* x_h, float* y_h, int n) {
	float* x_d, * y_d;
	cudaMalloc((void**)&x_d, n * sizeof(float));
	cudaMalloc((void**)&y_d, n * sizeof(float));
	cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(n/2);
	dim3 blockNum((n/2 + blockSize.x - 1) / blockSize.x);
	brent_kung_scan_kernel << <blockNum, blockSize, n * sizeof(float) >> > (x_d, y_d, n);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
}

void printArray(const char* name, const float* array, int size, int limit = 10) {
	std::cout << name << ": ";
	for (int i = 0; i < std::min(size, limit); i++)
		std::cout << array[i] << " ";
	std::cout << "..." << std::endl;
}
int main() {
	float* x_h = (float*)malloc(VECTOR_SIZE * sizeof(float));
	// 随机数生成器
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, 100.0f);
	for (int i = 0; i < VECTOR_SIZE; i++) {
		x_h[i] = dist(gen);
	}
	float* y_h_cpu = (float*)malloc(VECTOR_SIZE * sizeof(float));
	float* y_h_gpu = (float*)malloc(VECTOR_SIZE * sizeof(float));
	sequential_scan(x_h, y_h_cpu, VECTOR_SIZE);
	brent_kung_scan(x_h, y_h_gpu, VECTOR_SIZE);
	
	if (areFloatArrayEqual(y_h_cpu, y_h_gpu, VECTOR_SIZE))
		std::cout << "true"<<std::endl;
	else
		std::cout << "false"<<std::endl;
	printArray("ori", x_h, VECTOR_SIZE);
	printArray("cpu", y_h_cpu, VECTOR_SIZE);
	printArray("gpu", y_h_gpu, VECTOR_SIZE);
	free(x_h);
	free(y_h_cpu);
	free(y_h_gpu);
	return 0;
}