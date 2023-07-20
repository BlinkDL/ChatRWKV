#include "ATen/ATen.h"
#include <cuda_fp16.h>

template <typename T> T *data_ptr(torch::Tensor x) { return x.data_ptr<T>(); }
template <> inline half *data_ptr(torch::Tensor x) {
  return reinterpret_cast<half *>(x.data_ptr<at::Half>());
}
