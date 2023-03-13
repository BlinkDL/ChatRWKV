#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
typedef at::Half fp16;

void cuda_wkv_forward(int B, int T, int C, float *w, float *u, fp16 *k, fp16 *v, fp16 *y, float *aa, float *bb, float *pp);
void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  fp16 *y, int y_stride);
void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  fp16 *y);

void wkv_forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    cuda_wkv_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(), aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
}
void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    cuda_mm8_seq(
        B, N, M,
        x.data_ptr<fp16>(), x.stride(0),
        w.data_ptr<uint8_t>(), w.stride(0),
        mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
        my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
        y.data_ptr<fp16>(), y.stride(0));
}
void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    cuda_mm8_one(
        N, M,
        x.data_ptr<fp16>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
        my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
        y.data_ptr<fp16>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "wkv forward");
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_one", &mm8_one, "mm8 one");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
}
