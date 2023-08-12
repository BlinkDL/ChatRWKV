#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "element_wise.h"
#include "util.h"

namespace {
// Equivalent Python code:
// s1 = t_first * a + s
// s2 = a + t_decay * s
struct Fused1 {
  const float *t_first;
  const float *t_decay;
  const float *a;
  const float *s;
  const int32_t inner_size;
  /* out */ float *s1;
  /* out */ float *s2;

  __device__ void operator()(int i) const {
    const int j = i / inner_size;
    s1[i] = t_first[j] * a[i] + s[i];
    s2[i] = a[i] + t_decay[j] * s[i];
  }
};

/*
   Equivalent Python code:
   kx = xx * k_mix + sx * (1 - k_mix)
   vx = xx * v_mix + sx * (1 - v_mix)
   rx = xx * r_mix + sx * (1 - r_mix)
*/

struct Mix {
  const half *xx;
  const half *sx;
  const half *kvr_mix;
  const int stride;
  /* out */ half *kvrx;

  __device__ void operator()(int i) const {
    half xx_ = xx[i];
    half sx_ = sx[i];
    half k_mix_ = kvr_mix[i];
    half v_mix_ = kvr_mix[i + stride];
    half r_mix_ = kvr_mix[i + stride * 2];
    kvrx[i] = __hadd(__hmul(xx_, k_mix_),
                     __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    kvrx[i + stride] = __hadd(__hmul(xx_, v_mix_),
                              __hmul(sx_, __hsub(__float2half(1), v_mix_)));
    kvrx[i + stride * 2] = __hadd(__hmul(xx_, r_mix_),
                                  __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
};

struct ToHalf {
  const float *x;
  half *y;
  __device__ void operator()(int i) const { y[i] = __float2half(x[i]); }
};

struct InplaceAdd {
  __device__ __forceinline__ half operator()(int i) const {
    y[i] = __hadd(x[i], y[i]);
  }
  half *y;
  half *x;
};
} // namespace

using torch::Tensor;

void gemm_cublas_tensor(const Tensor &a, const Tensor &b, const Tensor &c);
void gemm_cublas(const void *a, const void *b, void *c, int batch, int ori_m,
                 int ori_n, int ori_k, at::ScalarType torch_input_dtype,
                 at::ScalarType torch_output_dtype);

Tensor att_one_v5(Tensor x, Tensor sx, Tensor s, Tensor ln_w, Tensor ln_b,
                  Tensor lx_w, Tensor lx_b, Tensor kvr_mix, Tensor kvrw,
                  Tensor ow, Tensor t_first, Tensor t_decay, Tensor tmp,
                  Tensor buf, /* out */ Tensor s2_t,
                  /* out */ Tensor x_plus_out_t) {
  const int x_numel = x.numel();
  Tensor xx = at::layer_norm(x, {x_numel}, ln_w, ln_b);
  int H = t_decay.size(0);
  int S = x_numel / H;
  char *buf_ptr = (char *)buf.data_ptr();
  half *kvrx = (half *)buf_ptr;
  float *kvr = (float *)(kvrx + 3 * x_numel);
  float *a = kvr + 3 * x_numel;
  half *tmp2 = (half *)(a + H * S * S);
  float *s1 = (float *)(tmp2 + x_numel);
  float *s2 = data_ptr<float>(s2_t);
  half *x_plus_out = data_ptr<half>(x_plus_out_t);

  element_wise(Mix{data_ptr<half>(xx), data_ptr<half>(sx),
                   data_ptr<half>(kvr_mix), static_cast<int>(x_numel), kvrx},
               x_numel);

  gemm_cublas(kvrx, data_ptr<half>(kvrw), kvr, 3, 1, x_numel, x_numel,
              at::kHalf, at::kFloat);
  float *k = kvr;
  float *v = k + x_numel;
  float *r = v + x_numel;

  gemm_cublas(k, v, a, H, S, S, 1, at::kFloat, at::kFloat);
  element_wise(Fused1{data_ptr<float>(t_first), data_ptr<float>(t_decay), a,
                      data_ptr<float>(s), static_cast<int32_t>(S * S), s1, s2},
               H * S * S);

  gemm_cublas(r, s1, data_ptr<float>(tmp), H, 1, S, S, at::kFloat, at::kFloat);
  tmp = at::group_norm(tmp, H, lx_w, lx_b);
  element_wise(ToHalf{data_ptr<float>(tmp), tmp2}, tmp.numel());

  gemm_cublas(tmp2, data_ptr<half>(ow), x_plus_out, 1, 1, x_numel, x_numel,
              at::kHalf, at::kHalf);
  element_wise(InplaceAdd{x_plus_out, data_ptr<half>(x)}, x.numel());
  return xx;
}
