#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// MatrixMulKernel  朴素版
__global__
void MatrixMulKernel(float* C_d, float* A_d, float* B_d, int M, int N, int K) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < N && row < M) {
		float sum = 0;
		for (int i = 0; i < K; i++) {
			sum += A_d[row * K + i] * B_d[i * N + col];
		}
		C_d[row * N + col] = sum;
	}
}
// MatrixMulKernel  考虑性能版
__global__
void matrixMulKernel(float* C_d, float* A_d, float* B_d, int M, int N, int K) {
	// 声明动态共享内存，一维的方式存放两个数组
	extern __shared__ float shared_mem[];
	//Ads放在前面，Bds放在后面
	// A的块大小：blockDim.y行 x blockDim.x列
    // B的块大小：blockDim.x行 x blockDim.y列
	float* Ads = shared_mem;
	float* Bds = shared_mem + blockDim.x * blockDim.y;
	// 计算当前线程索引
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// 计算
	float Pvalue = 0;
	// 计算需要处理的K维度的块数量
	int K_blocks = (K + blockDim.x - 1) / blockDim.x;
	for (int ph = 0; ph < K_blocks; ph++) {
		int a_col = ph * blockDim.x + tx; // A中的列索引
		if (row < M && (a_col < K))
			Ads[ty * blockDim.x + tx] = A_d[row * K + a_col];
		else
			Ads[ty * blockDim.x + tx] = 0.0f;
		int b_row = ph * blockDim.x + ty; // B中的行索引
		if (col < N && b_row < K)
			Bds[ty * blockDim.x + tx] = B_d[b_row * N + col];
		else
			Bds[ty * blockDim.x + tx] = 0.0f;
		__syncthreads();
		// 只使用块内实际可用的元素数量，可能小于blockDim.x
		int k_limit;
		if (K - ph * blockDim.x < blockDim.x)
			k_limit = K - ph * blockDim.x;
		else
			k_limit = blockDim.x;
		for (int k = 0; k < k_limit; k++) {
			Pvalue += Ads[ty * blockDim.x + k] * Bds[k * blockDim.x + tx];
		}
		__syncthreads();
	}
	if (row < M && col < N)
		C_d[row * N + col] = Pvalue;
}
__global__
void MatrixVectorMulKernel(float* C_d, float* A_d, float* B_d, int M) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < M) {
		float sum = 0;
		for (int i = 0; i < M; i++) {
			sum += A_d[row * M + i] * B_d[i];
		}
		C_d[row] = sum;
	}
}
void MatrixVectorMul(float* C_h, float* A_h, float* B_h, int M) {
	float* A_d, * B_d, * C_d;
	cudaMalloc((void**)&A_d, M * M * sizeof(float));
	cudaMalloc((void**)&B_d, M * sizeof(float));
	cudaMalloc((void**)&C_d, M * sizeof(float));
	cudaMemcpy(A_d, A_h, M * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, M * sizeof(float), cudaMemcpyHostToDevice);
	MatrixVectorMulKernel << <(M + 32 - 1) / 32, 32 >> > (C_d, A_d, B_d, M);
	cudaMemcpy(C_h, C_d, M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

// 内核入口
void MatrixMul(float* C_h, float* A_h, float* B_h, int M, int N, int K) {
	float* A_d, * B_d, * C_d;
	cudaMalloc((void**)&A_d, M * K * sizeof(float));
	cudaMalloc((void**)&B_d, K * N * sizeof(float));
	cudaMalloc((void**)&C_d, M * N * sizeof(float));
	cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
	size_t sharedMemSize = threadsPerBlock.x * threadsPerBlock.y * sizeof(float) * 2; // 两个数组的大小
	matrixMulKernel << <numBlocks, threadsPerBlock, sharedMemSize >> > (C_d, A_d, B_d, M, N, K);
	cudaMemcpy(C_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}
int main() {
	int M = 20, N = 30, K = 40;
	float* A_h = (float*)malloc(M * K * sizeof(float));
	float* B_h = (float*)malloc(K * N * sizeof(float));
	float* C_h = (float*)malloc(M * N * sizeof(float));
	// 初始化AB矩阵，随机数
	for (int i = 0; i < M * K; i++) {
		A_h[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for (int i = 0; i < K * N; i++) {
		B_h[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	MatrixMul(C_h, A_h, B_h, M, N, K);
	/*cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "设备名\t\t" << prop.name << std::endl;
	std::cout << "全局内存大小\t\t" << prop.totalGlobalMem/(1024*1024) <<"MB" << std::endl;
	std::cout << "总SM数\t\t" << prop.multiProcessorCount << std::endl;
	std::cout << "每块最大线程数\t\t" << prop.maxThreadsPerBlock << std::endl;
	std::cout << "每块的共享内存数\t" << prop.sharedMemPerBlock<<"B" << std::endl;
	std::cout << "每块的寄存器数\t\t" << prop.regsPerBlock << std::endl;      
	std::cout << std::endl;
	std::cout << "Warp大小\t\t" << prop.warpSize << std::endl;
	std::cout << "每个SM的最大线程数\t" << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大块数\t" << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大共享内存数\t" << prop.sharedMemPerMultiprocessor << "B" << std::endl;
	std::cout << "每个SM的最大寄存器数\t" << prop.regsPerMultiprocessor << std::endl;
	std::cout << std::endl;
	std::cout << "Max threads dim\t" << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
	std::cout << "Max grid size\t" << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
	*/
	free(A_h);
	free(B_h);
	free(C_h);
}