#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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
// template for caffe2::TypeMeta and c10::ScalarType
void gemm_cublas(const void *a, const void *b, void *c, int batch, int ori_m,
                 int ori_n, int ori_k, at::ScalarType torch_input_dtype,
                 at::ScalarType torch_output_dtype) {
  cublasHandle_t cublas_handle = get_cublas_handle();
  const auto cuda_input_dtype =
      torch_input_dtype == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
  const auto cuda_output_dtype =
      torch_output_dtype == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
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

#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
  const float sp_beta = 0.f;
  if (batch == 1) {
    CUBLAS_CHECK(cublasGemmEx(cublas_handle, cublas_trans_a, cublas_trans_b,
                              cublas_m, cublas_n, cublas_k, &sp_alpha, b,
                              cuda_input_dtype, cublas_lda, a, cuda_input_dtype,
                              cublas_ldb, &sp_beta, c, cuda_output_dtype,
                              cublas_ldc, compute_type, algo));
  } else {
    const long long int cublas_stride_a = cublas_m * cublas_k;
    const long long int cublas_stride_b = cublas_k * cublas_n;
    const long long int cublas_stride_c = cublas_m * cublas_n;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
        cublas_k, &sp_alpha, b, cuda_input_dtype, cublas_lda, cublas_stride_a,
        a, cuda_input_dtype, cublas_ldb, cublas_stride_b, &sp_beta, c,
        cuda_output_dtype, cublas_ldc, cublas_stride_c, batch, compute_type,
        algo));
  }
}

void gemm_cublas(const half *a, const half *b, float *c, int batch, int ori_m,
                 int ori_n, int ori_k) {
  return gemm_cublas(a, b, c, batch, ori_m, ori_n, ori_k, torch::kFloat16,
                     torch::kFloat32);
}

void gemm_cublas(const half *a, const half *b, half *c, int batch, int ori_m,
                 int ori_n, int ori_k) {
  return gemm_cublas(a, b, c, batch, ori_m, ori_n, ori_k, torch::kFloat16,
                     torch::kFloat16);
}

void gemm_cublas(const float *a, const float *b, float *c, int batch, int ori_m,
                 int ori_n, int ori_k) {
  return gemm_cublas(a, b, c, batch, ori_m, ori_n, ori_k, torch::kFloat32,
                     torch::kFloat32);
}

using torch::Tensor;
/*
  NOTE: blas gemm is column-major by default, but we need row-major output.
  The data of row-major, transposed matrix is exactly the same as the
  column-major, non-transposed matrix, and C = A * B ---> C^T = B^T * A^T
 */
void gemm_cublas_tensor(const Tensor &a, const Tensor &b, const Tensor &c) {
  if (a.sizes().size() == 1) {
    assert(b.sizes().size() == 2);
    return gemm_cublas(a.data_ptr(), b.data_ptr(), c.data_ptr(), 1, 1,
                       b.size(1), b.size(0), a.scalar_type(), c.scalar_type());
  } else if (a.sizes().size() == 3) {
    assert(b.sizes().size() == 3);
    return gemm_cublas(a.data_ptr(), b.data_ptr(), c.data_ptr(), a.size(0),
                       a.size(1), b.size(2), b.size(1), a.scalar_type(), c.scalar_type());
  } else {
    assert(a.sizes().size() == 2);
    assert(b.sizes().size() == 2);
    return gemm_cublas(a.data_ptr(), b.data_ptr(), c.data_ptr(), 1, a.size(0),
                       b.size(1), b.size(0), a.scalar_type(), c.scalar_type());
  }
}
