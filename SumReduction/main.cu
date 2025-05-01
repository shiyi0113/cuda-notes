#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <string>

__global__
void simpleSumReduction(float* output_d, float* input_d) {
    unsigned int index = 2 * threadIdx.x;
    for (size_t step = 1; step <= blockDim.x; step *= 2) {
        if (threadIdx.x % step == 0) {
            input_d[index] += input_d[index + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *output_d = input_d[0];
}

__global__
void convergentSumReduction(float* output_d, float* input_d) {
    unsigned int index = threadIdx.x;
    for (size_t step = blockDim.x; step >= 1; step /= 2) {
        if (threadIdx.x < step) {
            input_d[index] += input_d[index + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *output_d = input_d[0];
}

__global__
void sharedMemorySumReduction(float* output_d, float* input_d) {
    extern __shared__ float input_s[]; // 动态共享内存
    unsigned int index = threadIdx.x;
    input_s[index] = input_d[index] + input_d[index + blockDim.x];  // 第一轮计算结果存入共享内存
    for (size_t step = blockDim.x / 2; step >= 1; step /= 2) {
        __syncthreads();
        if (threadIdx.x < step) {
            input_s[index] += input_s[index + step];
        }
    }
    if (threadIdx.x == 0)
        *output_d = input_s[0];
}

__global__
void segmentMemorySumReduction(float* output_d, float* input_d) {
    extern __shared__ float input_s[]; // 动态共享内存
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int globalIndex = segment + threadIdx.x;
    input_s[threadIdx.x] = input_d[globalIndex] + input_d[globalIndex + blockDim.x];  // 第一轮计算结果存入共享内存
    for (size_t step = blockDim.x / 2; step >= 1; step /= 2) {
        __syncthreads();
        if (threadIdx.x < step) {
            input_s[threadIdx.x] += input_s[threadIdx.x + step];
        }
    }
    if (threadIdx.x == 0)
        atomicAdd(output_d, input_s[0]);
}

#define COARSE_FACTOR 2
__global__
void coarsenedMemorySumReduction(float* output_d, float* input_d) {
    extern __shared__ float input_s[]; // 动态共享内存
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int globalIndex = segment + threadIdx.x;
    float sum = input_d[globalIndex];
    for (size_t tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        sum += input_d[globalIndex + tile * blockDim.x];
    }
    input_s[threadIdx.x] = sum;
    for (size_t step = blockDim.x / 2; step >= 1; step /= 2) {
        __syncthreads();
        if (threadIdx.x < step) {
            input_s[threadIdx.x] += input_s[threadIdx.x + step];
        }
    }
    if (threadIdx.x == 0)
        atomicAdd(output_d, input_s[0]);
}

// 辅助函数：检查CUDA错误
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 测试函数：准备数据，执行内核，测量时间
double testKernel(void (*kernelLauncher)(float*, float*, int, int), float* h_input, int size, int numRuns) {
    float* d_input, * d_output, h_output;

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, sizeof(float)));

    // 拷贝输入数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    // 同步设备以确保准确计时
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 运行内核多次并测量时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA_ERROR(cudaMemset(d_output, 0, sizeof(float)));
        kernelLauncher(d_output, d_input, size, i);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // 获取结果
    CHECK_CUDA_ERROR(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    // 返回每次运行的平均时间（毫秒）
    return duration.count() / numRuns;
}

// 内核启动包装函数
void launchSimpleSumReduction(float* d_output, float* d_input, int size, int) {
    simpleSumReduction << <1, size / 2 >> > (d_output, d_input);
}

void launchConvergentSumReduction(float* d_output, float* d_input, int size, int) {
    convergentSumReduction << <1, size / 2 >> > (d_output, d_input);
}

void launchSharedMemorySumReduction(float* d_output, float* d_input, int size, int) {
    sharedMemorySumReduction << <1, size / 2, size / 2 * sizeof(float) >> > (d_output, d_input);
}

void launchSegmentMemorySumReduction(float* d_output, float* d_input, int size, int) {
    int blockSize = 256;
    int numBlocks = size / (2 * blockSize);
    segmentMemorySumReduction << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_output, d_input);
}

void launchCoarsenedMemorySumReduction(float* d_output, float* d_input, int size, int) {
    int blockSize = 256;
    int numBlocks = size / (2 * COARSE_FACTOR * blockSize);
    coarsenedMemorySumReduction << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_output, d_input);
}

// 主函数
int main() {
    // 测试参数
    const int sizes[] = { 1024, 8192, 65536, 524288, 2097152 };
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    const int numRuns = 100; // 每个内核运行的次数

    // 输出表头
    std::cout << std::setw(12) << "Size"
        << std::setw(20) << "Simple (ms)"
        << std::setw(20) << "Convergent (ms)"
        << std::setw(20) << "Shared Mem (ms)"
        << std::setw(20) << "Segment Mem (ms)"
        << std::setw(20) << "Coarsened Mem (ms)" << std::endl;
    std::cout << std::string(112, '-') << std::endl;

    // 对每个数据大小进行测试
    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];

        // 准备输入数据
        std::vector<float> h_input(size);
        for (int j = 0; j < size; j++) {
            h_input[j] = 1.0f; // 使用简单的值以便验证结果
        }

        // 测试各个内核
        double simpleTime = 0, convergentTime = 0, sharedMemTime = 0, segmentMemTime = 0, coarsenedMemTime = 0;

        // 只有当数据大小足够小时才测试简单方法和收敛方法（它们只使用一个block）
        if (size <= 65536) {
            simpleTime = testKernel(launchSimpleSumReduction, h_input.data(), size, numRuns);
            convergentTime = testKernel(launchConvergentSumReduction, h_input.data(), size, numRuns);
            sharedMemTime = testKernel(launchSharedMemorySumReduction, h_input.data(), size, numRuns);
        }
        else {
            simpleTime = -1;
            convergentTime = -1;
            sharedMemTime = -1;
        }

        // 这些方法可以处理更大的数据
        segmentMemTime = testKernel(launchSegmentMemorySumReduction, h_input.data(), size, numRuns);
        coarsenedMemTime = testKernel(launchCoarsenedMemorySumReduction, h_input.data(), size, numRuns);

        // 输出结果
        std::cout << std::setw(12) << size;
        std::cout << std::setw(20) << (simpleTime < 0 ? "N/A" : std::to_string(simpleTime));
        std::cout << std::setw(20) << (convergentTime < 0 ? "N/A" : std::to_string(convergentTime));
        std::cout << std::setw(20) << (sharedMemTime < 0 ? "N/A" : std::to_string(sharedMemTime));
        std::cout << std::setw(20) << segmentMemTime;
        std::cout << std::setw(20) << coarsenedMemTime << std::endl;
    }

    return 0;
}