#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 定义块大小
#define BLOCK_SIZE 16

// 基础矩阵乘法内核（不使用共享内存）
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 使用共享内存的优化矩阵乘法内核
__global__ void matrixMulSharedKernel(float* A, float* B, float* C, int M, int N, int K) {
    // 为每个块分配共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前线程要处理的C中的元素位置
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // 循环遍历所有分块
    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        // 将A和B的分块加载到共享内存
        if (row < M && i * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + i * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (i * BLOCK_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // 等待块内所有线程完成数据加载
        __syncthreads();
        
        // 计算分块内的点积
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // 等待所有线程完成计算后再加载下一个分块
        __syncthreads();
    }
    
    // 将结果写入C矩阵
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 主机函数：执行矩阵乘法
void matrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K, bool useSharedMemory = true) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 在设备上分配内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // 定义网格和块维度
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动内核
    if (useSharedMemory) {
        matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    
    // 检查内核执行是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // 等待所有设备操作完成
    cudaDeviceSynchronize();
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 验证函数：使用CPU计算矩阵乘法进行验证
void matrixMultiplyCPU(float* A, float* B, float* C, int M, int N, int K) {
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

// 测试函数
int main() {
    // 定义矩阵维度
    const int M = 512;  // A的行数，C的行数
    const int K = 256;  // A的列数，B的行数
    const int N = 512;  // B的列数，C的列数
    
    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);
    
    // 初始化矩阵A和B
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    std::cout << "Matrix dimensions: A[" << M << "x" << K << "], B[" << K << "x" << N << "], C[" << M << "x" << N << "]" << std::endl;
    
    // 使用GPU计算（共享内存版本）
    std::cout << "Running GPU matrix multiplication with shared memory..." << std::endl;
    matrixMultiply(h_A.data(), h_B.data(), h_C_gpu.data(), M, N, K, true);
    
    // 使用CPU计算进行验证
    std::cout << "Running CPU matrix multiplication for verification..." << std::endl;
    matrixMultiplyCPU(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    
    // 验证结果
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    
    std::cout << "Maximum error between GPU and CPU results: " << maxError << std::endl;
    
    // 测试不使用共享内存的版本
    std::cout << "\nRunning GPU matrix multiplication without shared memory..." << std::endl;
    matrixMultiply(h_A.data(), h_B.data(), h_C_gpu.data(), M, N, K, false);
    
    // 再次验证
    maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    
    std::cout << "Maximum error between GPU (no shared memory) and CPU results: " << maxError << std::endl;
    
    return 0;
}