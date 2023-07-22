#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUBLAS_CHECK(condition)                                             \
  for (cublasStatus_t _cublas_check_status = (condition);                   \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                      \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_cublas_check_status) +         \
                             " at " + std::to_string(__LINE__));

#define CUDA_CHECK(condition)                                               \
  for (cudaError_t _cuda_check_status = (condition);                        \
       _cuda_check_status != cudaSuccess;)                                  \
    throw std::runtime_error(                                                  \
        "CUDA error " +                                                        \
        std::string(cudaGetErrorString(_cuda_check_status)) + " at " +      \
        std::to_string(__LINE__));

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
void gemm_fp16_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  const auto cuda_data_type = CUDA_R_16F;
  const auto cuda_c_data_type =
      c.dtype() == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
  const auto compute_type = CUDA_R_32F;
  const float sp_alpha = 1.f;
  // swap a and b, and use CUBLAS_OP_N. see the notes above
  std::swap(a, b);
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  // m = (B^T).size(0) = B.size(1), and = A.size(1) after swap
  const int m = a.size(1);
  const int k = a.size(0);
  const int n = b.size(0);
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
  CUBLAS_CHECK(cublasGemmEx(
      cublas_handle, cublas_trans_a, cublas_trans_b, m, n, k, &sp_alpha,
      a.data_ptr(), cuda_data_type, cublas_lda, b.data_ptr(), cuda_data_type,
      cublas_ldb, &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc,
      compute_type, algo));
}
