#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CPU�汾�������ӷ�
void vecAddCPU(float* A, float* B, float* C, int n) {
    // ִ�������ӷ�
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// CUDA�˺�������GPU��ִ�������ӷ�
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    // ���㵱ǰ�̵߳�ȫ������
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {  // ȷ������������߽�
        C[i] = A[i] + B[i];
    }
}
// GPU�汾�������ӷ�
void vecAddGPU(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float* A_d, * B_d, * C_d;
    // �����豸�ڴ�GPU
    //cudaMalloc((void**)&A_d, size);
    //cudaMalloc((void**)&B_d, size);
    //cudaMalloc((void**)&C_d, size);
    // �����豸�ڴ� �쳣���
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
    // �������ݵ��豸   Ŀ�ĵأ�Դ����С����ô�� �������������豸������Դ������A_h��->Ŀ�ĵأ��豸A_d��
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // ���ú˺�����������
    float threadsPerBlock = 256.0;

    // �����˺���
    vecAddKernel << <ceil(n / threadsPerBlock), threadsPerBlock >> > (A_d, B_d, C_d, n);

    // ���ƽ��������
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    // �ͷ��豸�ڴ�
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int vectorSize = 10000000;
    size_t size = vectorSize * sizeof(float);
    // ���������ڴ�CPU
    float* A_h = (float*)malloc(size);
    float* B_h = (float*)malloc(size);
    float* C_gpu = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);

    // ��ʼ����������
    srand(time(0));  //���������
    for (int i = 0; i < vectorSize; i++) {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
    }
    vecAddGPU(A_h, B_h, C_gpu, vectorSize);
    vecAddCPU(A_h, B_h, C_cpu, vectorSize);
    // �ͷ������ڴ�
    free(A_h);
    free(B_h);
    free(C_gpu);
    free(C_cpu);

    return 0;
}