#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CPU版本的向量加法
void vecAddCPU(float* A, float* B, float* C, int n) {
    // 执行向量加法
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// CUDA核函数，在GPU上执行向量加法
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    // 计算当前线程的全局索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {  // 确保不超出数组边界
        C[i] = A[i] + B[i];
    }
}
// GPU版本的向量加法
void vecAddGPU(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float* A_d, * B_d, * C_d;
    // 分配设备内存GPU
    //cudaMalloc((void**)&A_d, size);
    //cudaMalloc((void**)&B_d, size);
    //cudaMalloc((void**)&C_d, size);
    // 分配设备内存 异常检查
    cudaError_t erra = cudaMalloc((void**)&A_d, size);
    if (erra != cudaSuccess) {
        std::cerr << cudaGetErrorString(erra) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaError_t errb = cudaMalloc((void**)&B_d, size);
    if (errb != cudaSuccess) {
        std::cerr << cudaGetErrorString(errb) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaError_t errc = cudaMalloc((void**)&C_d, size);
    if (errc != cudaSuccess) {
        std::cerr << cudaGetErrorString(errc) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
    // 复制数据到设备   目的地，源，大小，怎么走 这里是主机到设备，所以源（主机A_h）->目的地（设备A_d）
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // 配置核函数启动参数
    float threadsPerBlock = 256.0;

    // 启动核函数
    vecAddKernel << <ceil(n / threadsPerBlock), threadsPerBlock >> > (A_d, B_d, C_d, n);

    // 复制结果到主机
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    // 释放设备内存
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int vectorSize = 10000000;
    size_t size = vectorSize * sizeof(float);
    // 分配主机内存CPU
    float* A_h = (float*)malloc(size);
    float* B_h = (float*)malloc(size);
    float* C_gpu = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);

    // 初始化输入向量
    srand(time(0));  //随机数种子
    for (int i = 0; i < vectorSize; i++) {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
    }
    vecAddGPU(A_h, B_h, C_gpu, vectorSize);
    vecAddCPU(A_h, B_h, C_cpu, vectorSize);
    // 释放主机内存
    free(A_h);
    free(B_h);
    free(C_gpu);
    free(C_cpu);

    return 0;
}