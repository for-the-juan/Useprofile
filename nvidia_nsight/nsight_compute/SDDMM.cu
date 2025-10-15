#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// SDDMM内核：计算 (A * B) ⊙ S，其中⊙是逐元素乘法
// A: M×K, B: K×N, S: M×N (稀疏矩阵，COO格式)
// 结果存储在C: M×N (稀疏矩阵，与S相同的稀疏模式)
template<typename T>
__global__ void sddmmKernel(
    const T* __restrict__ A,        // 稠密矩阵 A [M×K]
    const T* __restrict__ B,        // 稠密矩阵 B [K×N] 
    const int* __restrict__ row_idx, // 稀疏矩阵S的行索引 [nnz]
    const int* __restrict__ col_idx, // 稀疏矩阵S的列索引 [nnz]
    const T* __restrict__ S_values, // 稀疏矩阵S的值 [nnz] (可选)
    T* __restrict__ output,         // 输出值 [nnz]
    int M, int N, int K, int nnz,   // 矩阵维度
    bool has_S_values = false) {    // 是否使用S的值
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        int row = row_idx[idx];
        int col = col_idx[idx];
        
        // 计算A的第row行和B的第col列的点积
        T sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // 如果提供了S的值，进行逐元素乘法
        if (has_S_values) {
            output[idx] = sum * S_values[idx];
        } else {
            output[idx] = sum;
        }
    }
}

// 优化版本：使用共享内存缓存数据块
template<typename T, int BLOCK_SIZE>
__global__ void sddmmOptimizedKernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const int* __restrict__ row_idx,
    const int* __restrict__ col_idx,
    const T* __restrict__ S_values,
    T* __restrict__ output,
    int M, int N, int K, int nnz,
    bool has_S_values = false) {
    
    __shared__ T A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    
    if (idx >= nnz) return;
    
    int row = row_idx[idx];
    int col = col_idx[idx];
    
    T sum = 0.0;
    
    // 分块处理内积计算
    for (int k_block = 0; k_block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k_block++) {
        int k_A = k_block * BLOCK_SIZE + thread_id;
        int k_B = k_block * BLOCK_SIZE + thread_id;
        
        // 将A的块加载到共享内存
        if (row < M && k_A < K) {
            A_shared[thread_id][thread_id] = A[row * K + k_A];
        } else {
            A_shared[thread_id][thread_id] = 0.0;
        }
        
        // 将B的块加载到共享内存  
        if (k_B < K && col < N) {
            B_shared[thread_id][thread_id] = B[k_B * N + col];
        } else {
            B_shared[thread_id][thread_id] = 0.0;
        }
        
        __syncthreads();
        
        // 计算当前块的内积贡献
        int k_end = min(BLOCK_SIZE, K - k_block * BLOCK_SIZE);
        for (int k = 0; k < k_end; k++) {
            sum += A_shared[thread_id][k] * B_shared[k][thread_id];
        }
        
        __syncthreads();
    }
    
    // 应用稀疏矩阵的值（如果提供）
    if (has_S_values) {
        output[idx] = sum * S_values[idx];
    } else {
        output[idx] = sum;
    }
}

// 针对排序数据的优化版本（行优先排序）
template<typename T>
__global__ void sddmmSortedKernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const int* __restrict__ row_idx,
    const int* __restrict__ col_idx,
    const T* __restrict__ S_values,
    T* __restrict__ output,
    int M, int N, int K, int nnz,
    bool has_S_values = false) {
    
    int row = blockIdx.x;
    int lane_id = threadIdx.x;
    
    if (row >= M) return;
    
    // 找到当前行的非零元素范围
    // 注意：这个版本假设数据已经按行排序
    int start_idx = -1, end_idx = -1;
    
    // 简化的查找 - 实际中应该使用更高效的方法
    for (int i = 0; i < nnz; i++) {
        if (row_idx[i] == row) {
            if (start_idx == -1) start_idx = i;
            end_idx = i;
        } else if (row_idx[i] > row && start_idx != -1) {
            break;
        }
    }
    
    if (start_idx == -1) return; // 这一行没有非零元素
    
    // 处理当前行的所有非零元素
    for (int idx = start_idx + lane_id; idx <= end_idx; idx += blockDim.x) {
        if (idx <= end_idx && row_idx[idx] == row) {
            int col = col_idx[idx];
            
            T sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            
            if (has_S_values) {
                output[idx] = sum * S_values[idx];
            } else {
                output[idx] = sum;
            }
        }
    }
}

// COO格式稀疏矩阵结构
template<typename T>
struct SparseMatrixCOO {
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<T> values;
    int rows, cols, nnz;
    
    SparseMatrixCOO(int m, int n, int nz) : rows(m), cols(n), nnz(nz) {
        row_indices.resize(nz);
        col_indices.resize(nz);
        values.resize(nz);
    }
};

// 生成随机稀疏矩阵（COO格式）
template<typename T>
SparseMatrixCOO<T> generateRandomSparseMatrix(int M, int N, float density) {
    int total_elements = M * N;
    int nnz = static_cast<int>(total_elements * density);
    
    SparseMatrixCOO<T> sparse_mat(M, N, nnz);
    
    // 生成随机非零位置
    for (int i = 0; i < nnz; i++) {
        sparse_mat.row_indices[i] = rand() % M;
        sparse_mat.col_indices[i] = rand() % N;
        sparse_mat.values[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    return sparse_mat;
}

// 生成随机稠密矩阵
template<typename T>
std::vector<T> generateRandomDenseMatrix(int rows, int cols) {
    std::vector<T> matrix(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    return matrix;
}

// SDDMM主机函数
template<typename T>
void sddmm(
    const std::vector<T>& A,           // 稠密矩阵 A [M×K]
    const std::vector<T>& B,           // 稠密矩阵 B [K×N]
    const SparseMatrixCOO<T>& S,       // 稀疏矩阵 S [M×N]
    std::vector<T>& output,            // 输出值 [nnz]
    int M, int N, int K,
    bool use_optimized = false,
    bool use_sorted = false) {
    
    // 验证维度
    if (A.size() != M * K || B.size() != K * N || output.size() != S.nnz) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return;
    }
    
    // 分配设备内存
    T *d_A, *d_B, *d_S_values, *d_output;
    int *d_row_idx, *d_col_idx;
    
    cudaMalloc(&d_A, M * K * sizeof(T));
    cudaMalloc(&d_B, K * N * sizeof(T));
    cudaMalloc(&d_S_values, S.nnz * sizeof(T));
    cudaMalloc(&d_output, S.nnz * sizeof(T));
    cudaMalloc(&d_row_idx, S.nnz * sizeof(int));
    cudaMalloc(&d_col_idx, S.nnz * sizeof(int));
    
    // 复制数据到设备
    cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_values, S.values.data(), S.nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_idx, S.row_indices.data(), S.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, S.col_indices.data(), S.nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    // 配置内核参数
    const int BLOCK_SIZE = 256;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((S.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动内核
    if (use_sorted) {
        // 使用排序版本（按行分组）
        dim3 sorted_blockDim(256);
        dim3 sorted_gridDim(M);
        sddmmSortedKernel<T><<<sorted_gridDim, sorted_blockDim>>>(
            d_A, d_B, d_row_idx, d_col_idx, d_S_values, d_output,
            M, N, K, S.nnz, true);
    } else if (use_optimized) {
        // 使用优化版本（共享内存）
        sddmmOptimizedKernel<T, 16><<<gridDim, blockDim>>>(
            d_A, d_B, d_row_idx, d_col_idx, d_S_values, d_output,
            M, N, K, S.nnz, true);
    } else {
        // 使用基础版本
        sddmmKernel<T><<<gridDim, blockDim>>>(
            d_A, d_B, d_row_idx, d_col_idx, d_S_values, d_output,
            M, N, K, S.nnz, true);
    }
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(output.data(), d_output, S.nnz * sizeof(T), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_S_values);
    cudaFree(d_output);
    cudaFree(d_row_idx);
    cudaFree(d_col_idx);
}

// CPU参考实现
template<typename T>
void sddmmCPU(
    const std::vector<T>& A,
    const std::vector<T>& B,
    const SparseMatrixCOO<T>& S,
    std::vector<T>& output) {
    
    for (int idx = 0; idx < S.nnz; idx++) {
        int row = S.row_indices[idx];
        int col = S.col_indices[idx];
        
        T sum = 0.0;
        for (int k = 0; k < A.size() / S.rows; k++) { // 假设K = A.size()/M
            sum += A[row * (A.size() / S.rows) + k] * B[k * B.size() / (A.size() / S.rows) + col];
        }
        
        output[idx] = sum * S.values[idx];
    }
}

// 测试函数
int main() {
    // 定义矩阵维度
    const int M = 512;  // A的行数，S的行数
    const int K = 256;  // A的列数，B的行数
    const int N = 512;  // B的列数，S的列数
    const float density = 0.01f; // 稀疏矩阵密度
    
    std::cout << "Matrix dimensions: A[" << M << "x" << K << "], B[" << K << "x" << N 
              << "], S[" << M << "x" << N << "] with density " << density << std::endl;
    
    // 生成测试数据
    auto A = generateRandomDenseMatrix<float>(M, K);
    auto B = generateRandomDenseMatrix<float>(K, N);
    auto S = generateRandomSparseMatrix<float>(M, N, density);
    
    std::vector<float> output_gpu(S.nnz);
    std::vector<float> output_cpu(S.nnz);
    
    std::cout << "Non-zero elements: " << S.nnz << std::endl;
    
    // CPU参考计算
    std::cout << "Running CPU reference..." << std::endl;
    sddmmCPU(A, B, S, output_cpu);
    
    // GPU基础版本
    std::cout << "Running GPU baseline..." << std::endl;
    sddmm(A, B, S, output_gpu, M, N, K, false, false);
    
    // 验证结果
    float max_error = 0.0f;
    for (int i = 0; i < S.nnz; i++) {
        float error = fabs(output_gpu[i] - output_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    std::cout << "Baseline max error: " << max_error << std::endl;
    
    // GPU优化版本
    std::cout << "Running GPU optimized..." << std::endl;
    sddmm(A, B, S, output_gpu, M, N, K, true, false);
    
    // 验证优化版本结果
    max_error = 0.0f;
    for (int i = 0; i < S.nnz; i++) {
        float error = fabs(output_gpu[i] - output_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    std::cout << "Optimized max error: " << max_error << std::endl;
    
    return 0;
}