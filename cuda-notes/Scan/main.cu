#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cub/cub.cuh>
#define VECTOR_SIZE 4096

void printArray(const char* name, const float* array, int size, int limit = 10) {
	std::cout << name << ": ";
	for (int i = 0; i < std::min(size, limit); i++)
		std::cout << array[i] << " ";
	std::cout << "..." << std::endl;
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
void kogge_stone_scan_double_buffer_kernel(float* x_d, float* y_d, int n) {
	extern __shared__ float buffer[];
	float* XY0 = buffer;
	float* XY1 = buffer + blockDim.x;

	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	// 数据先加载到缓冲区0
	if (i < n) {
		XY0[threadIdx.x] = x_d[i];
	}
	else {
		XY0[threadIdx.x] = 0.0f;
	}
	bool flag = true;
	for (size_t step = 1; step < blockDim.x; step *= 2) {
		__syncthreads();
		if (flag) {
			if (threadIdx.x >= step)
				XY1[threadIdx.x] = XY0[threadIdx.x] + XY0[threadIdx.x - step];
			else
				XY1[threadIdx.x] = XY0[threadIdx.x];
			flag = false;
		}
		else {
			if (threadIdx.x >= step)
				XY0[threadIdx.x] = XY1[threadIdx.x] + XY1[threadIdx.x - step];
			else
				XY0[threadIdx.x] = XY1[threadIdx.x];
			flag = true;
		}
	}
	__syncthreads();
	if (i < n) {
		if(flag)
			y_d[i] = XY0[threadIdx.x];
		else
			y_d[i] = XY1[threadIdx.x];
	}
}
void kogge_stone_scan_double_buffer(float* x_h, float* y_h, int n) {
	float* x_d, * y_d;
	cudaMalloc((void**)&x_d, n * sizeof(float));
	cudaMalloc((void**)&y_d, n * sizeof(float));
	cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(n);
	dim3 blockNum((n + blockSize.x - 1) / blockSize.x);
	kogge_stone_scan_double_buffer_kernel << <blockNum, blockSize, 2 * blockSize.x * sizeof(float) >> > (x_d, y_d, n);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
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
/*Thread coarsening*/
__global__
void three_phase_parallel_inclusive_scan_kernel(float* x_d, float* y_d, int n,int shared_mem_size) {
	extern __shared__ float XY[];
	// 同一个线程块中，一个线程处理多个数据
	int start_pos = blockIdx.x * shared_mem_size;
	int tx = threadIdx.x;
	// 把所有数据从全局内存搬运到共享内存中
	for (int j = tx; j < shared_mem_size; j += blockDim.x) {
		XY[j] = start_pos + j < n ? x_d[start_pos + j] : 0.0f;
	}
	__syncthreads();
	// 阶段一：块内顺序扫描
	int coarse_size = shared_mem_size / blockDim.x;
	int start = coarse_size * tx;
	int stop = start + coarse_size;
	float temp = 0.0f;
	if (start_pos + start < n) {    // 最后一块可能无法分全
		for (int i = start; i < stop; i++) {
			temp += XY[i];
			XY[i] = temp;
		}
	}
	__syncthreads();
	// 阶段二：块间并列扫描  使用Brent-Kung算法
	// 对每一块的最后一个元素进行并列扫描
	for (size_t step = 1; step < blockDim.x; step *= 2) {
		size_t index = (tx + 1) * 2 * step * coarse_size - 1;
		if (index < shared_mem_size)
			XY[index] += XY[index - step * coarse_size];
		__syncthreads();
	}
	for (size_t step = shared_mem_size/4; step >=coarse_size; step /= 2) {
		size_t index = (tx + 1) * 2 * step - 1;
		if (index + step < shared_mem_size)
			XY[index + step] += XY[index];
		__syncthreads();
	}
	// 阶段三：将每一块的最后一个元素加到下一块的前三个元素中
	if (tx != 0) {
		float value = XY[start - 1];
		for (int i = start; i < stop - 1; i++) {
			XY[i] += value;
		}
	}
	__syncthreads();
	//返回结果到全局内存中
	for (int i = tx; i < shared_mem_size; i += blockDim.x) {
		if (start_pos + i < n) {
			y_d[start_pos + i] = XY[i];
		}
	}
}
void three_phase_parallel_inclusive_scan(float* x_h, float* y_h, int n) {
	float* x_d, * y_d;
	cudaMalloc((void**)&x_d, n * sizeof(float));
	cudaMalloc((void**)&y_d, n * sizeof(float));
	cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(1024);
	dim3 blockNum((n / 4 + blockSize.x - 1) / blockSize.x);
	three_phase_parallel_inclusive_scan_kernel << <blockNum.x, blockSize.x, n * sizeof(float) >> > (x_d, y_d,n, n);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
}

//cuda_cub
void cub_inclusive_scan(float* x_h, float* y_h, int n) {
	float* x_d, * y_d;
	cudaMalloc((void**)&x_d, n * sizeof(float));
	cudaMalloc((void**)&y_d, n * sizeof(float));
	cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, x_d, y_d, n);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, x_d, y_d, n);
	cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_temp_storage);
	cudaFree(x_d);
	cudaFree(y_d);
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

	float* y_h_cub = (float*)malloc(VECTOR_SIZE * sizeof(float));
	float* y_h_gpu = (float*)malloc(VECTOR_SIZE * sizeof(float));

	three_phase_parallel_inclusive_scan(x_h, y_h_gpu, VECTOR_SIZE);
	cub_inclusive_scan(x_h, y_h_cub, VECTOR_SIZE);
	if (areFloatArrayEqual(y_h_cub, y_h_gpu, VECTOR_SIZE))
		std::cout << "true"<<std::endl;
	else
		std::cout << "false"<<std::endl;
	printArray("src", x_h, VECTOR_SIZE);
	printArray("cub", y_h_cub, VECTOR_SIZE);
	printArray("gpu", y_h_gpu, VECTOR_SIZE);
	free(x_h);
	free(y_h_cub);
	free(y_h_gpu);
	return 0;
}