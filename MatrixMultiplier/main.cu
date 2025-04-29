#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

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
void MatrixMul(float* C_h, float* A_h, float* B_h, int M, int N, int K) {
	float* A_d, * B_d, * C_d;
	cudaMalloc((void**)&A_d, M * K * sizeof(float));
	cudaMalloc((void**)&B_d, K * N * sizeof(float));
	cudaMalloc((void**)&C_d, M * N * sizeof(float));
	cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
	MatrixMulKernel << <numBlocks, threadsPerBlock >> > (C_d, A_d, B_d, M, N, K);
	cudaMemcpy(C_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
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
	free(A_h);
	free(B_h);
	free(C_h);
}