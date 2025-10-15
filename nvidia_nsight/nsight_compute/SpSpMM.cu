#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

// CSR格式稀疏矩阵
template<typename T>
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;
    
    CSRMatrix(int m = 0, int n = 0, int nz = 0) : rows(m), cols(n), nnz(nz) {
        row_ptr.resize(rows + 1, 0);
        if (nz > 0) {
            col_idx.resize(nnz);
            values.resize(nnz);
        }
    }
};

// COO格式稀疏矩阵
template<typename T>
struct COOMatrix {
    int rows, cols, nnz;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<T> values;
    
    COOMatrix(int m = 0, int n = 0, int nz = 0) : rows(m), cols(n), nnz(nz) {
        if (nz > 0) {
            row_idx.resize(nnz);
            col_idx.resize(nnz);
            values.resize(nnz);
        }
    }
};

// 将COO矩阵转换为CSR格式
template<typename T>
CSRMatrix<T> coo_to_csr(const COOMatrix<T>& coo) {
    CSRMatrix<T> csr(coo.rows, coo.cols, coo.nnz);
    
    // 计算每行的非零元素数量
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_idx[i];
        csr.row_ptr[row + 1]++;
    }
    
    // 计算行指针（前缀和）
    for (int i = 1; i <= coo.rows; i++) {
        csr.row_ptr[i] += csr.row_ptr[i - 1];
    }
    
    // 填充列索引和值
    std::vector<int> next_pos(coo.rows, 0);
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_idx[i];
        int pos = csr.row_ptr[row] + next_pos[row];
        csr.col_idx[pos] = coo.col_idx[i];
        csr.values[pos] = coo.values[i];
        next_pos[row]++;
    }
    
    return csr;
}

// 生成随机稀疏矩阵（COO格式）
template<typename T>
COOMatrix<T> generateRandomSparseMatrix(int rows, int cols, float density) {
    int total_elements = rows * cols;
    int nnz = std::max(1, static_cast<int>(total_elements * density));
    
    COOMatrix<T> coo(rows, cols, nnz);
    
    // 生成非零元素位置
    std::vector<std::pair<int, int>> positions;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (static_cast<float>(rand()) / RAND_MAX < density) {
                positions.push_back({i, j});
            }
        }
    }
    
    // 如果生成的元素太少，添加一些随机元素
    while (positions.size() < nnz) {
        int i = rand() % rows;
        int j = rand() % cols;
        positions.push_back({i, j});
    }
    
    // 如果生成的元素太多，随机选择一部分
    if (positions.size() > nnz) {
        std::random_shuffle(positions.begin(), positions.end());
        positions.resize(nnz);
    }
    
    // 填充数据
    for (int i = 0; i < nnz; i++) {
        coo.row_idx[i] = positions[i].first;
        coo.col_idx[i] = positions[i].second;
        coo.values[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    return coo;
}

// 第一步：计算每行非零元素数量的内核
__global__ void computeRowNnzKernel(
    const int* A_row_ptr, const int* A_col_idx,
    const int* B_row_ptr, const int* B_col_idx,
    int* C_row_nnz, int A_rows, int B_cols) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;
    
    // 使用位图记录当前行出现的列
    const int BITMAP_SIZE = 1024; // 假设列数不超过1024
    __shared__ unsigned int bitmap[BITMAP_SIZE / 32];
    
    // 初始化位图
    for (int i = threadIdx.x; i < BITMAP_SIZE / 32; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();
    
    int start_A = A_row_ptr[row];
    int end_A = A_row_ptr[row + 1];
    
    int nnz_count = 0;
    
    // 遍历A的当前行的所有非零元素
    for (int i = start_A; i < end_A; i++) {
        int col_A = A_col_idx[i];
        
        // 遍历B的对应行的所有非零元素
        int start_B = B_row_ptr[col_A];
        int end_B = B_row_ptr[col_A + 1];
        
        for (int j = start_B; j < end_B; j++) {
            int col_B = B_col_idx[j];
            
            // 使用位图记录列
            int word = col_B / 32;
            int bit = col_B % 32;
            
            if (word < BITMAP_SIZE / 32) {
                unsigned int old = atomicOr(&bitmap[word], 1u << bit);
                if (!(old & (1u << bit))) {
                    nnz_count++;
                }
            }
        }
    }
    
    C_row_nnz[row] = nnz_count;
}

// 第二步：实际计算SpGEMM的内核
__global__ void spgemmComputeKernel(
    const int* A_row_ptr, const int* A_col_idx, const float* A_values,
    const int* B_row_ptr, const int* B_col_idx, const float* B_values,
    const int* C_row_ptr, int* C_col_idx, float* C_values,
    int A_rows, int B_cols) {
    
    int row = blockIdx.x;
    if (row >= A_rows) return;
    
    extern __shared__ float shared_mem[];
    int* col_buffer = (int*)shared_mem;
    float* val_buffer = (float*)&col_buffer[B_cols];
    
    // 初始化共享内存缓冲区
    for (int i = threadIdx.x; i < B_cols; i += blockDim.x) {
        col_buffer[i] = -1;
        val_buffer[i] = 0.0f;
    }
    __syncthreads();
    
    int start_A = A_row_ptr[row];
    int end_A = A_row_ptr[row + 1];
    
    // 遍历A的当前行的所有非零元素
    for (int i = start_A; i < end_A; i++) {
        int col_A = A_col_idx[i];
        float val_A = A_values[i];
        
        // 遍历B的对应行的所有非零元素
        int start_B = B_row_ptr[col_A];
        int end_B = B_row_ptr[col_A + 1];
        
        for (int j = start_B; j < end_B; j++) {
            int col_B = B_col_idx[j];
            float val_B = B_values[j];
            
            // 累加到缓冲区
            atomicAdd(&val_buffer[col_B], val_A * val_B);
            col_buffer[col_B] = col_B; // 标记该列有非零元素
        }
    }
    __syncthreads();
    
    // 将结果写入全局内存
    int start_C = C_row_ptr[row];
    int count = 0;
    
    for (int col = threadIdx.x; col < B_cols; col += blockDim.x) {
        if (col_buffer[col] != -1 && val_buffer[col] != 0.0f) {
            int pos = start_C + count;
            C_col_idx[pos] = col;
            C_values[pos] = val_buffer[col];
            count++;
        }
    }
}

// 使用thrust的SpGEMM实现（更稳定）
template<typename T>
CSRMatrix<T> spgemmThrust(const CSRMatrix<T>& A, const CSRMatrix<T>& B) {
    int M = A.rows;
    int N = B.cols;
    
    // 步骤1：计算每行的非零元素数量
    thrust::device_vector<int> d_A_row_ptr(A.row_ptr.begin(), A.row_ptr.end());
    thrust::device_vector<int> d_A_col_idx(A.col_idx.begin(), A.col_idx.end());
    thrust::device_vector<T> d_A_values(A.values.begin(), A.values.end());
    
    thrust::device_vector<int> d_B_row_ptr(B.row_ptr.begin(), B.row_ptr.end());
    thrust::device_vector<int> d_B_col_idx(B.col_idx.begin(), B.col_idx.end());
    thrust::device_vector<T> d_B_values(B.values.begin(), B.values.end());
    
    thrust::device_vector<int> d_C_row_nnz(M);
    
    // 调用内核计算每行非零元素数量
    dim3 blockDim(256);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
    
    computeRowNnzKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_A_row_ptr.data()),
        thrust::raw_pointer_cast(d_A_col_idx.data()),
        thrust::raw_pointer_cast(d_B_row_ptr.data()),
        thrust::raw_pointer_cast(d_B_col_idx.data()),
        thrust::raw_pointer_cast(d_C_row_nnz.data()),
        M, N);
    
    cudaDeviceSynchronize();
    
    // 步骤2：计算行指针（前缀和）
    thrust::device_vector<int> d_C_row_ptr(M + 1);
    thrust::exclusive_scan(d_C_row_nnz.begin(), d_C_row_nnz.end(), d_C_row_ptr.begin());
    
    // 获取总非零元素数量
    int total_nnz;
    thrust::copy(d_C_row_ptr.begin() + M, d_C_row_ptr.begin() + M + 1, &total_nnz);
    thrust::copy(d_C_row_nnz.begin() + M - 1, d_C_row_nnz.end(), &total_nnz);
    total_nnz += d_C_row_ptr[M - 1];
    
    // 步骤3：分配输出矩阵内存
    thrust::device_vector<int> d_C_col_idx(total_nnz);
    thrust::device_vector<T> d_C_values(total_nnz);
    
    // 步骤4：实际计算SpGEMM
    size_t shared_mem_size = (N * sizeof(int) + N * sizeof(T));
    spgemmComputeKernel<<<M, 256, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_A_row_ptr.data()),
        thrust::raw_pointer_cast(d_A_col_idx.data()),
        thrust::raw_pointer_cast(d_A_values.data()),
        thrust::raw_pointer_cast(d_B_row_ptr.data()),
        thrust::raw_pointer_cast(d_B_col_idx.data()),
        thrust::raw_pointer_cast(d_B_values.data()),
        thrust::raw_pointer_cast(d_C_row_ptr.data()),
        thrust::raw_pointer_cast(d_C_col_idx.data()),
        thrust::raw_pointer_cast(d_C_values.data()),
        M, N);
    
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    CSRMatrix<T> C(M, N, total_nnz);
    thrust::copy(d_C_row_ptr.begin(), d_C_row_ptr.end(), C.row_ptr.begin());
    thrust::copy(d_C_col_idx.begin(), d_C_col_idx.end(), C.col_idx.begin());
    thrust::copy(d_C_values.begin(), d_C_values.end(), C.values.begin());
    
    return C;
}

// 简单的CPU SpGEMM参考实现
template<typename T>
CSRMatrix<T> spgemmCPU(const CSRMatrix<T>& A, const CSRMatrix<T>& B) {
    int M = A.rows;
    int N = B.cols;
    
    // 使用哈希表存储中间结果
    std::vector<std::unordered_map<int, T>> temp_result(M);
    
    for (int i = 0; i < M; i++) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            int col_A = A.col_idx[j];
            T val_A = A.values[j];
            
            for (int k = B.row_ptr[col_A]; k < B.row_ptr[col_A + 1]; k++) {
                int col_B = B.col_idx[k];
                T val_B = B.values[k];
                temp_result[i][col_B] += val_A * val_B;
            }
        }
    }
    
    // 计算总非零元素数量和行指针
    int total_nnz = 0;
    std::vector<int> row_ptr(M + 1, 0);
    
    for (int i = 0; i < M; i++) {
        row_ptr[i + 1] = row_ptr[i] + temp_result[i].size();
        total_nnz += temp_result[i].size();
    }
    
    // 创建输出矩阵
    CSRMatrix<T> C(M, N, total_nnz);
    C.row_ptr = row_ptr;
    
    // 填充列索引和值
    std::vector<int> col_idx(total_nnz);
    std::vector<T> values(total_nnz);
    
    for (int i = 0, idx = 0; i < M; i++) {
        for (const auto& entry : temp_result[i]) {
            col_idx[idx] = entry.first;
            values[idx] = entry.second;
            idx++;
        }
    }
    
    C.col_idx = col_idx;
    C.values = values;
    
    return C;
}

// 验证两个CSR矩阵是否相等（允许浮点误差）
template<typename T>
bool validateCSR(const CSRMatrix<T>& A, const CSRMatrix<T>& B, T tolerance = 1e-5) {
    if (A.rows != B.rows || A.cols != B.cols || A.nnz != B.nnz) {
        std::cout << "Dimension or NNZ mismatch!" << std::endl;
        return false;
    }
    
    for (int i = 0; i <= A.rows; i++) {
        if (A.row_ptr[i] != B.row_ptr[i]) {
            std::cout << "Row pointer mismatch at row " << i << std::endl;
            return false;
        }
    }
    
    // 由于非零元素的顺序可能不同，我们需要按行和列排序后比较
    for (int i = 0; i < A.rows; i++) {
        int start = A.row_ptr[i];
        int end = A.row_ptr[i + 1];
        
        // 收集当前行的所有元素
        std::vector<std::pair<int, T>> row_A, row_B;
        for (int j = start; j < end; j++) {
            row_A.push_back({A.col_idx[j], A.values[j]});
            row_B.push_back({B.col_idx[j], B.values[j]});
        }
        
        // 按列索引排序
        std::sort(row_A.begin(), row_A.end());
        std::sort(row_B.begin(), row_B.end());
        
        // 比较排序后的结果
        for (size_t j = 0; j < row_A.size(); j++) {
            if (row_A[j].first != row_B[j].first || 
                std::abs(row_A[j].second - row_B[j].second) > tolerance) {
                std::cout << "Value mismatch at (" << i << ", " << row_A[j].first 
                          << "): " << row_A[j].second << " vs " << row_B[j].second << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

// 打印CSR矩阵信息
template<typename T>
void printCSRInfo(const CSRMatrix<T>& mat, const std::string& name) {
    std::cout << name << ": " << mat.rows << " x " << mat.cols 
              << " with " << mat.nnz << " non-zero elements" << std::endl;
}

int main() {
    // 设置随机种子
    srand(42);
    
    // 定义矩阵维度
    const int M = 128;  // A的行数
    const int K = 64;   // A的列数，B的行数
    const int N = 96;   // B的列数
    const float density = 0.05f; // 稀疏矩阵密度
    
    std::cout << "SpGEMM Test: A[" << M << "x" << K << "] * B[" << K << "x" << N << "]" << std::endl;
    std::cout << "Density: " << density << std::endl;
    
    // 生成随机稀疏矩阵
    std::cout << "Generating random sparse matrices..." << std::endl;
    auto coo_A = generateRandomSparseMatrix<float>(M, K, density);
    auto coo_B = generateRandomSparseMatrix<float>(K, N, density);
    
    // 转换为CSR格式
    auto csr_A = coo_to_csr(coo_A);
    auto csr_B = coo_to_csr(coo_B);
    
    printCSRInfo(csr_A, "Matrix A");
    printCSRInfo(csr_B, "Matrix B");
    
    // CPU参考计算
    std::cout << "Running CPU reference SpGEMM..." << std::endl;
    auto csr_C_cpu = spgemmCPU(csr_A, csr_B);
    printCSRInfo(csr_C_cpu, "CPU Result");
    
    // GPU计算
    std::cout << "Running GPU SpGEMM..." << std::endl;
    auto csr_C_gpu = spgemmThrust(csr_A, csr_B);
    printCSRInfo(csr_C_gpu, "GPU Result");
    
    // 验证结果
    std::cout << "Validating results..." << std::endl;
    bool is_valid = validateCSR(csr_C_cpu, csr_C_gpu, 1e-4f);
    
    if (is_valid) {
        std::cout << "SUCCESS: GPU results match CPU reference!" << std::endl;
    } else {
        std::cout << "FAILURE: GPU results do not match CPU reference!" << std::endl;
    }
    
    // 性能测试
    std::cout << "\nPerformance test with larger matrices..." << std::endl;
    
    const int M_large = 512;
    const int K_large = 256;
    const int N_large = 512;
    
    auto coo_A_large = generateRandomSparseMatrix<float>(M_large, K_large, 0.02f);
    auto coo_B_large = generateRandomSparseMatrix<float>(K_large, N_large, 0.02f);
    
    auto csr_A_large = coo_to_csr(coo_A_large);
    auto csr_B_large = coo_to_csr(coo_B_large);
    
    printCSRInfo(csr_A_large, "Large Matrix A");
    printCSRInfo(csr_B_large, "Large Matrix B");
    
    // 只测试GPU性能
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    auto csr_C_large = spgemmThrust(csr_A_large, csr_B_large);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printCSRInfo(csr_C_large, "Large Result");
    std::cout << "GPU computation time: " << milliseconds << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}