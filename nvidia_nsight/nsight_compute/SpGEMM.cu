#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// CSR格式的稀疏矩阵与稠密矩阵乘法
__global__ void spmmCSRKernel(const float* __restrict__ values,
                             const int* __restrict__ col_indices,
                             const int* __restrict__ row_ptrs,
                             const float* __restrict__ dense_matrix,
                             float* __restrict__ output_matrix,
                             int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int elem = row_start; elem < row_end; elem++) {
                int col_idx = col_indices[elem];
                float sparse_val = values[elem];
                float dense_val = dense_matrix[col_idx * N + col];
                sum += sparse_val * dense_val;
            }
            output_matrix[row * N + col] = sum;
        }
    }
}

// 优化版本：使用共享内存和向量化
__global__ void spmmCSROptimizedKernel(const float* __restrict__ values,
                                      const int* __restrict__ col_indices,
                                      const int* __restrict__ row_ptrs,
                                      const float* __restrict__ dense_matrix,
                                      float* __restrict__ output_matrix,
                                      int M, int K, int N) {
    extern __shared__ float shared_dense[];
    
    int row = blockIdx.x;
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_in_warp = threadIdx.x % 32;
    
    if (row >= M) return;
    
    int row_start = row_ptrs[row];
    int row_end = row_ptrs[row + 1];
    int row_length = row_end - row_start;
    
    // 每个warp处理输出矩阵的一列
    for (int col_start = 0; col_start < N; col_start += blockDim.x) {
        int col = col_start + lane_id;
        
        float sum = 0.0f;
        
        // 处理稀疏行的所有非零元素
        for (int elem = row_start; elem < row_end; elem++) {
            int col_idx = col_indices[elem];
            float sparse_val = values[elem];
            
            if (col < N) {
                float dense_val = dense_matrix[col_idx * N + col];
                sum += sparse_val * dense_val;
            }
        }
        
        if (col < N) {
            output_matrix[row * N + col] = sum;
        }
    }
}

// COO格式的稀疏矩阵与稠密矩阵乘法
__global__ void spmmCOOKernel(const float* __restrict__ values,
                             const int* __restrict__ row_indices,
                             const int* __restrict__ col_indices,
                             const float* __restrict__ dense_matrix,
                             float* __restrict__ output_matrix,
                             int nnz, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        float val = values[idx];
        
        for (int j = 0; j < N; j++) {
            atomicAdd(&output_matrix[row * N + j], val * dense_matrix[col * N + j]);
        }
    }
}

// 稀疏矩阵结构（CSR格式）
struct SparseMatrixCSR {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols, nnz;
};

// 将稠密矩阵转换为CSR格式
SparseMatrixCSR denseToCSR(const std::vector<float>& dense_matrix, int rows, int cols, float epsilon = 1e-6f) {
    SparseMatrixCSR csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.row_ptrs.push_back(0);
    
    for (int i = 0; i < rows; i++) {
        int row_nnz = 0;
        for (int j = 0; j < cols; j++) {
            float val = dense_matrix[i * cols + j];
            if (fabs(val) > epsilon) {
                csr.values.push_back(val);
                csr.col_indices.push_back(j);
                row_nnz++;
            }
        }
        csr.row_ptrs.push_back(csr.row_ptrs.back() + row_nnz);
    }
    
    csr.nnz = csr.values.size();
    return csr;
}

// 主机函数：执行稀疏-稠密矩阵乘法
void sparseDenseMultiply(const SparseMatrixCSR& sparse_mat,
                        const std::vector<float>& dense_matrix,
                        std::vector<float>& output_matrix,
                        int M, int K, int N,
                        bool useOptimized = false) {
    // 验证维度
    if (sparse_mat.cols != K || dense_matrix.size() != K * N) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return;
    }
    
    // 分配设备内存
    float *d_values, *d_dense, *d_output;
    int *d_col_indices, *d_row_ptrs;
    
    cudaMalloc(&d_values, sparse_mat.nnz * sizeof(float));
    cudaMalloc(&d_col_indices, sparse_mat.nnz * sizeof(int));
    cudaMalloc(&d_row_ptrs, (sparse_mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_dense, K * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    
    // 初始化输出矩阵为0
    cudaMemset(d_output, 0, M * N * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_values, sparse_mat.values.data(), sparse_mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, sparse_mat.col_indices.data(), sparse_mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptrs, sparse_mat.row_ptrs.data(), (sparse_mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense, dense_matrix.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置内核参数
    dim3 blockDim(256);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
    
    // 启动内核
    if (useOptimized) {
        // 优化版本可能需要调整块大小
        spmmCSROptimizedKernel<<<M, 256>>>(d_values, d_col_indices, d_row_ptrs, d_dense, d_output, M, K, N);
    } else {
        spmmCSRKernel<<<gridDim, blockDim>>>(d_values, d_col_indices, d_row_ptrs, d_dense, d_output, M, K, N);
    }
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(output_matrix.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptrs);
    cudaFree(d_dense);
    cudaFree(d_output);
}

// CPU参考实现
void sparseDenseMultiplyCPU(const SparseMatrixCSR& sparse_mat,
                           const std::vector<float>& dense_matrix,
                           std::vector<float>& output_matrix,
                           int M, int K, int N) {
    std::fill(output_matrix.begin(), output_matrix.end(), 0.0f);
    
    for (int i = 0; i < M; i++) {
        int row_start = sparse_mat.row_ptrs[i];
        int row_end = sparse_mat.row_ptrs[i + 1];
        
        for (int elem = row_start; elem < row_end; elem++) {
            int col_idx = sparse_mat.col_indices[elem];
            float sparse_val = sparse_mat.values[elem];
            
            for (int j = 0; j < N; j++) {
                output_matrix[i * N + j] += sparse_val * dense_matrix[col_idx * N + j];
            }
        }
    }
}

// 测试函数
int main() {
    // 定义矩阵维度
    const int M = 1024;  // 稀疏矩阵行数
    const int K = 512;   // 稀疏矩阵列数，稠密矩阵行数
    const int N = 256;   // 稠密矩阵列数
    
    // 生成随机稀疏矩阵（5% 密度）
    std::vector<float> sparse_dense(M * K, 0.0f);
    int nnz_count = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (static_cast<float>(rand()) / RAND_MAX < 0.05f) { // 5% 非零元素
                sparse_dense[i * K + j] = static_cast<float>(rand()) / RAND_MAX;
                nnz_count++;
            }
        }
    }
    
    // 转换为CSR格式
    SparseMatrixCSR sparse_csr = denseToCSR(sparse_dense, M, K);
    
    // 生成随机稠密矩阵
    std::vector<float> dense_matrix(K * N);
    for (int i = 0; i < K * N; i++) {
        dense_matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // 输出矩阵
    std::vector<float> output_gpu(M * N);
    std::vector<float> output_cpu(M * N);
    
    std::cout << "Sparse matrix: " << M << " x " << K << " with " << sparse_csr.nnz 
              << " non-zero elements (" << (100.0 * sparse_csr.nnz) / (M * K) << "% density)" << std::endl;
    std::cout << "Dense matrix: " << K << " x " << N << std::endl;
    std::cout << "Output matrix: " << M << " x " << N << std::endl;
    
    // GPU计算
    std::cout << "Running GPU sparse-dense matrix multiplication..." << std::endl;
    sparseDenseMultiply(sparse_csr, dense_matrix, output_gpu, M, K, N, false);
    
    // CPU计算验证
    std::cout << "Running CPU reference calculation..." << std::endl;
    sparseDenseMultiplyCPU(sparse_csr, dense_matrix, output_cpu, M, K, N);
    
    // 验证结果
    float max_error = 0.0f;
    float max_value = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(output_gpu[i] - output_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (fabs(output_cpu[i]) > max_value) {
            max_value = fabs(output_cpu[i]);
        }
    }
    
    std::cout << "Maximum absolute error: " << max_error << std::endl;
    std::cout << "Maximum relative error: " << (max_value > 0 ? max_error / max_value : 0) << std::endl;
    
    // 测试优化版本
    std::cout << "\nRunning optimized GPU version..." << std::endl;
    sparseDenseMultiply(sparse_csr, dense_matrix, output_gpu, M, K, N, true);
    
    // 验证优化版本结果
    max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(output_gpu[i] - output_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    
    std::cout << "Maximum absolute error (optimized): " << max_error << std::endl;
    std::cout << "Maximum relative error (optimized): " << (max_value > 0 ? max_error / max_value : 0) << std::endl;
    
    return 0;
}