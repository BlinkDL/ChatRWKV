#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <torch/extension.h>

template <typename T> T *data_ptr(torch::Tensor x) { return x.data_ptr<T>(); }
template <> inline half *data_ptr(torch::Tensor x) {
  return reinterpret_cast<half *>(x.data_ptr<at::Half>());
}

inline __host__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float4 b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z),
                     fmaxf(a.w, b.w));
}
inline __host__ __device__ float4 expf(float4 a) {
  return make_float4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
}
