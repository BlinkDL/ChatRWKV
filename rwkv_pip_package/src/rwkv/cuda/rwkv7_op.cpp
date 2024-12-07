#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::Half fp16;
typedef at::BFloat16 bf16;
typedef float fp32;

void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y);
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *w, fp16 *k, fp16 *v, fp16 *a, fp16 *b, fp16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *w, fp32 *k, fp32 *v, fp32 *a, fp32 *b, fp32 *y);

void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), w.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), a.data_ptr<bf16>(), b.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), w.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), a.data_ptr<fp16>(), b.data_ptr<fp16>(), y.data_ptr<fp16>());
}
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), w.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), a.data_ptr<fp32>(), b.data_ptr<fp32>(), y.data_ptr<fp32>());
}

TORCH_LIBRARY(wkv7s, m) {
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp16", forward_fp16);
    m.def("forward_fp32", forward_fp32);
}
