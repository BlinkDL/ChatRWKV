#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _cublas_check_status = (condition);                      \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                         \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_cublas_check_status) + " at " +   \
                             std::to_string(__LINE__));

#define CUDA_CHECK(condition)                                                  \
  for (cudaError_t _cuda_check_status = (condition);                           \
       _cuda_check_status != cudaSuccess;)                                     \
    throw std::runtime_error(                                                  \
        "CUDA error " + std::string(cudaGetErrorString(_cuda_check_status)) +  \
        " at " + std::to_string(__LINE__));

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t cublas_handle = []() {
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
#if CUDA_VERSION < 11000
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#else
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif // CUDA_VERSION < 11000
    return handle;
  }();
  return cublas_handle;
}

/*
  NOTE: blas gemm is column-major by default, but we need row-major output.
  The data of row-major, transposed matrix is exactly the same as the
  column-major, non-transposed matrix, and C = A * B ---> C^T = B^T * A^T
 */
void gemm_fp16_cublas(const void *a, const void *b, void *c, int ori_m,
                      int ori_n, int ori_k, bool output_fp32) {
  const auto cuda_data_type = CUDA_R_16F;
  const auto cuda_c_data_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
  const auto compute_type = CUDA_R_32F;
  const float sp_alpha = 1.f;
  // use CUBLAS_OP_N. see the notes above
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  // m = (B^T).size(0) = B.size(1) = n;
  const int cublas_m = ori_n;
  const int cublas_k = ori_k;
  // comptiable with rwkv one mode, where 1-D tensor * 2-D tensor
  // const int n = a.dense_dim() == 1 ? 1 : a.size(0);
  const int cublas_n = ori_m;
  const int cublas_lda = cublas_m;
  const int cublas_ldb = cublas_k;
  const int cublas_ldc = cublas_m;
  cublasHandle_t cublas_handle = get_cublas_handle();

#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
  const float sp_beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(
      cublas_handle, cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
      cublas_k, &sp_alpha, b, cuda_data_type, cublas_lda,
      a, cuda_data_type, cublas_ldb, &sp_beta, c,
      cuda_c_data_type, cublas_ldc, compute_type, algo));
}

/*
  NOTE: blas gemm is column-major by default, but we need row-major output.
  The data of row-major, transposed matrix is exactly the same as the
  column-major, non-transposed matrix, and C = A * B ---> C^T = B^T * A^T
 */
void gemm_fp16_cublas_tensor(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  if (a.sizes().size() == 1) {
    assert(b.sizes().size() == 2);
    a = at::unsqueeze(a, 0);
  }
  const auto cuda_data_type = CUDA_R_16F;
  const auto cuda_c_data_type =
      c.dtype() == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
  const auto compute_type = CUDA_R_32F;
  const float sp_alpha = 1.f;
  // swap a and b, and use CUBLAS_OP_N. see the notes above
  std::swap(a, b);
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  // m = (B^T).size(0) = B.size(1), and = A.size(1) after swap,
  // negative axis is used because of the existence of batch matmul.
  const int m = a.size(-1);
  const int k = a.size(-2);
  const int n = b.size(-2);
  const int cublas_lda = m;
  const int cublas_ldb = k;
  const int cublas_ldc = m;
  cublasHandle_t cublas_handle = get_cublas_handle();

#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
  const float sp_beta = 0.f;
  if (a.sizes().size() == 2 && b.sizes().size() == 2) {
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, m, n, k, &sp_alpha,
        a.data_ptr(), cuda_data_type, cublas_lda, b.data_ptr(), cuda_data_type,
        cublas_ldb, &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc,
        compute_type, algo));
  } else {
    // batch matmul
    assert(a.sizes().size() == 3 && b.sizes().size() == 3);

    const long long int cublas_stride_a = m * k;
    const long long int cublas_stride_b = k * n;
    const long long int cublas_stride_c = m * n;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, m,
        n, k, &sp_alpha, a.data_ptr(), cuda_data_type, cublas_lda,
        cublas_stride_a, b.data_ptr(), cuda_data_type, cublas_ldb, cublas_stride_b,
        &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc, cublas_stride_c,
        a.size(0), compute_type, algo));
  }
}
