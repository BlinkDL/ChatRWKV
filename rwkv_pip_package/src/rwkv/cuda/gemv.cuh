// Based on https://github.com/wangsiping97/FastGEMV

#include "util.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

__device__ __forceinline__ float warp_reduce_sum(float sum,
                                                 unsigned int thread_num) {
  if (thread_num >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
  if (thread_num >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
  if (thread_num >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
  if (thread_num >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
  if (thread_num >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <typename OutputT, typename Func>
__global__ void _gemv_fp16(half *mat, half *vec, OutputT *res, unsigned int n,
                           unsigned int num_per_thread, const Func epilogue) {
  static_assert(std::is_same_v<OutputT, float> || std::is_same_v<OutputT, half>,
                "Output type must be float or half");
  {
    float sum = 0;
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;
    float4 *mat4 = reinterpret_cast<float4 *>(mat);
    float4 *vec4 = reinterpret_cast<float4 *>(vec);

#pragma unroll
    for (int iter = 0; iter < num_per_thread >> 3; iter++) {
      unsigned int j = start_idx + iter * blockDim.x;
      if (j < n >> 3) {
        float4 vec_val = vec4[j];
        float4 mat_val = mat4[row * (n >> 3) + j];
        const half2 *vec_h1 = (half2 *)&vec_val.x;
        const half2 *vec_h2 = (half2 *)&vec_val.y;
        const half2 *vec_h3 = (half2 *)&vec_val.z;
        const half2 *vec_h4 = (half2 *)&vec_val.w;
        const half2 *mat_h1 = (half2 *)&mat_val.x;
        const half2 *mat_h2 = (half2 *)&mat_val.y;
        const half2 *mat_h3 = (half2 *)&mat_val.z;
        const half2 *mat_h4 = (half2 *)&mat_val.w;
        sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
        sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
        sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
        sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
        sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
        sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
        sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
        sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
      }
    }

    sum = warp_reduce_sum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
      if (tid == 0) {
        if constexpr (std::is_same_v<OutputT, float>) {
          res[row] = epilogue(sum, row);
        } else {
          res[row] = epilogue(__float2half(sum), row);
        }
      }
      return;
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    if (laneId == 0)
      warpLevelSums[threadIdx.y][warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE)
              ? warpLevelSums[threadIdx.y][laneId]
              : 0.0;
    // Final reduce using first warp
    if (warpId == 0)
      sum = warp_reduce_sum(sum, blockDim.x / WARP_SIZE);
    if (tid == 0) {
      if constexpr (std::is_same_v<OutputT, float>) {
        res[row] = epilogue(sum, row);
      } else {
        res[row] = epilogue(__float2half(sum), row);
      }
    }
  }
}

template <typename OutputT, typename Func>
void gemv_fp16(torch::Tensor mat, torch::Tensor vec, torch::Tensor out,
               const Func &func) {
  const int32_t BLOCK_DIM_X = 32;
  const int32_t BLOCK_DIM_Y = 16;
  assert(BLOCK_DIM_Y <= SHARED_MEM_MAX_ROWS);
  assert(BLOCK_DIM_X * BLOCK_DIM_Y <= MAX_THREADS_PER_BLOCK);

  auto N = mat.size(0);
  auto K = mat.size(1);
  assert(vec.size(0) == K);
  assert(out.size(0) == N);
  const int32_t num_per_thread = K / BLOCK_DIM_X;
  assert(num_per_thread >= 8);

  dim3 grid_dim(1, N / BLOCK_DIM_Y);
  dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
  _gemv_fp16<OutputT><<<grid_dim, block_dim>>>(
      data_ptr<half>(mat), data_ptr<half>(vec), data_ptr<OutputT>(out), K,
      num_per_thread, func);
}

template <typename T> struct IdentityEpilogue {
  __device__ __forceinline__ T operator()(T x, unsigned int idx) const {
    return x;
  }
};

struct SigmoidEpilogue {
  __device__ __forceinline__ half operator()(half x, unsigned int idx) const {
    return hrcp(__hadd(__float2half(1.0), hexp(__hneg(x))));
  }
};

struct ReLUAndSqaureEpilogue {
  __device__ __forceinline__ half operator()(half x, unsigned int idx) const {
    return __hgt(x, __float2half(0.0)) ? __hmul(x, x) : __float2half(0.0);
  }
};

struct BiasEpilogue {
  BiasEpilogue(half *bias) : bias(bias) {}
  __device__ __forceinline__ half operator()(half x, unsigned int idx) const {
    return __hadd(bias[idx], x);
  }
  half *bias;
};

struct ScaleAndBiasEpilogue {
  ScaleAndBiasEpilogue(half *scale, half *bias) : scale(scale), bias(bias) {}
  __device__ __forceinline__ half operator()(half x, unsigned int idx) const {
    return __hadd(bias[idx], __hmul(scale[idx], x));
  }
  half *scale;
  half *bias;
};
