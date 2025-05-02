# cuda-notes

This repository contains code written during my journey learning CUDA.

## Introduction

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA that enables developers to use CUDA-enabled GPUs to significantly accelerate computing applications. This repository documents the code I've written while learning CUDA. My learning is based on the textbook "Programming Massively Parallel Processors", a classic GPU programming textbook co-authored by NVIDIA's Chief Scientist David B. Kirk and Wen-mei W. Hwu, which comprehensively covers parallel programming techniques and optimization methods for modern GPUs.

## My Environment

- CUDA Toolkit: 12.8
- GPU: NVIDIA GeForce RTX 4060

### RTX 4060 Hardware Specifications
- Warp size: 32
- Number of Streaming Multiprocessors (SMs): 24
- Maximum threads per block: 1024
- Maximum threads per SM: 1536
- Maximum blocks per SM: 24
- Maximum shared memory per SM: 102400B
- Maximum registers per SM: 65536

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Samples Repository](https://github.com/NVIDIA/cuda-samples)
- 《Programming Massively Parallel Processors》 - David B. Kirk & Wen-mei W. Hwu

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details