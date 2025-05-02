#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

// MatrixMulKernel 朴素版
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

// MatrixMulKernel 考虑性能版
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

// CPU版本矩阵乘法，用于结果验证
void MatrixMulCPU(float* C, float* A, float* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 使用朴素版内核
void MatrixMulSimple(float* C_h, float* A_h, float* B_h, int M, int N, int K) {
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

// 使用优化版内核
void MatrixMulOptimized(float* C_h, float* A_h, float* B_h, int M, int N, int K) {
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

// 比较两个矩阵是否相等（考虑浮点误差）
bool CompareMatrix(float* A, float* B, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(A[i] - B[i]) > epsilon) {
            std::cout << "不匹配位置: " << i << " 值: " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 打印矩阵（用于调试）
void PrintMatrix(float* M, int rows, int cols, const char* name) {
    std::cout << "矩阵 " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, 5); i++) {
        for (int j = 0; j < std::min(cols, 5); j++) {
            std::cout << std::fixed << std::setprecision(4) << M[i * cols + j] << " ";
        }
        std::cout << (cols > 5 ? "..." : "") << std::endl;
    }
    if (rows > 5) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

// 创建特定类型的测试矩阵
void CreateTestMatrix(float* M, int rows, int cols, int testType) {
    switch (testType) {
    case 0: // 随机矩阵
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; i++) {
            M[i] = dis(gen);
        }
    }
    break;
    case 1: // 单位矩阵
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                M[i * cols + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        break;
    case 2: // 全1矩阵
        for (int i = 0; i < rows * cols; i++) {
            M[i] = 1.0f;
        }
        break;
    case 3: // 对角矩阵
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                M[i * cols + j] = (i == j) ? static_cast<float>(i + 1) : 0.0f;
            }
        }
        break;
    }
}

// 运行测试用例
bool RunTest(int M, int N, int K, int testTypeA, int testTypeB, int kernelType, bool printDetails = false) {
    std::cout << "====== 测试 " << (kernelType == 0 ? "朴素版" : "优化版") << " ======" << std::endl;
    std::cout << "矩阵大小: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;

    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C_gpu = new float[M * N];
    float* C_cpu = new float[M * N];

    // 创建测试矩阵
    CreateTestMatrix(A, M, K, testTypeA);
    CreateTestMatrix(B, K, N, testTypeB);

    if (printDetails) {
        PrintMatrix(A, M, K, "A");
        PrintMatrix(B, K, N, "B");
    }

    // 运行GPU版本
    if (kernelType == 0) {
        MatrixMulSimple(C_gpu, A, B, M, N, K);
    }
    else {
        MatrixMulOptimized(C_gpu, A, B, M, N, K);
    }

    // 运行CPU参考版本
    MatrixMulCPU(C_cpu, A, B, M, N, K);

    if (printDetails) {
        PrintMatrix(C_gpu, M, N, "C_gpu");
        PrintMatrix(C_cpu, M, N, "C_cpu");
    }

    // 比较结果
    bool result = CompareMatrix(C_gpu, C_cpu, M * N);
    if (result) {
        std::cout << "测试通过！GPU和CPU结果匹配。" << std::endl;
    }
    else {
        std::cout << "测试失败！GPU和CPU结果不匹配。" << std::endl;
    }
    std::cout << std::endl;

    delete[] A;
    delete[] B;
    delete[] C_gpu;
    delete[] C_cpu;

    return result;
}

int main() {
    // 测试用例类型说明
    std::cout << "矩阵类型说明：" << std::endl;
    std::cout << "0: 随机矩阵" << std::endl;
    std::cout << "1: 单位矩阵" << std::endl;
    std::cout << "2: 全1矩阵" << std::endl;
    std::cout << "3: 对角矩阵" << std::endl << std::endl;

    bool allTestsPassed = true;

    // 测试用例1：小矩阵，随机数据
    allTestsPassed &= RunTest(4, 4, 4, 0, 0, 0, true);
    allTestsPassed &= RunTest(4, 4, 4, 0, 0, 1, true);

    // 测试用例2：单位矩阵乘法（结果应该与原矩阵相同）
    allTestsPassed &= RunTest(10, 10, 10, 0, 1, 0);
    allTestsPassed &= RunTest(10, 10, 10, 0, 1, 1);

    // 测试用例3：全1矩阵乘法
    allTestsPassed &= RunTest(8, 8, 8, 2, 2, 0);
    allTestsPassed &= RunTest(8, 8, 8, 2, 2, 1);

    // 测试用例4：非方形矩阵
    allTestsPassed &= RunTest(10, 20, 30, 0, 0, 0);
    allTestsPassed &= RunTest(10, 20, 30, 0, 0, 1);

    // 测试用例5：较大矩阵
    allTestsPassed &= RunTest(64, 64, 64, 0, 0, 0);
    allTestsPassed &= RunTest(64, 64, 64, 0, 0, 1);

    // 测试用例6：边界情况-维度不是block大小的整数倍
    allTestsPassed &= RunTest(7, 13, 19, 0, 0, 0);
    allTestsPassed &= RunTest(7, 13, 19, 0, 0, 1);

    // 测试用例7：对角矩阵乘法
    allTestsPassed &= RunTest(10, 10, 10, 3, 3, 0);
    allTestsPassed &= RunTest(10, 10, 10, 3, 3, 1);

    if (allTestsPassed) {
        std::cout << "所有测试用例通过！两个矩阵乘法内核均正确。" << std::endl;
    }
    else {
        std::cout << "部分测试用例失败，请检查内核实现。" << std::endl;
    }

    return 0;
}