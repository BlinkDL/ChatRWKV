#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "element_wise.h"
#include "util.h"

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
  const half *k_mix;
  const half *v_mix;
  const half *r_mix;
  /* out */ half *kx;
  /* out */ half *vx;
  /* out */ half *rx;

  __device__ void operator()(int i) const {
    half xx_ = xx[i];
    half sx_ = sx[i];
    half k_mix_ = k_mix[i];
    half v_mix_ = v_mix[i];
    half r_mix_ = r_mix[i];
    kx[i] = __hadd(__hmul(xx_, k_mix_),
                   __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    vx[i] = __hadd(__hmul(xx_, v_mix_),
                   __hmul(sx_, __hsub(__float2half(1), v_mix_)));
    rx[i] = __hadd(__hmul(xx_, r_mix_),
                   __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
};

using torch::Tensor;

void gemm_fp16_cublas_tensor(Tensor a, Tensor b, Tensor c);

Tensor att_one_v5(Tensor x, Tensor sx, Tensor s, Tensor ln_w, Tensor ln_b,
                  Tensor lx_w, Tensor lx_b, Tensor k_mix, Tensor v_mix,
                  Tensor r_mix, Tensor kw,
                  /* imm */ Tensor kx, Tensor vw, /* imm */ Tensor vx,
                  Tensor rw,
                  /* imm */ Tensor rx, Tensor ow, Tensor t_first,
                  /* imm */ Tensor k, Tensor t_decay, /* imm */ Tensor v,
                  /* imm */ Tensor r, /* imm */ Tensor s1,
                  /* out */ Tensor x_plus_out, /* out */ Tensor s2) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  element_wise(Mix{data_ptr<half>(xx), data_ptr<half>(sx),
                   data_ptr<half>(k_mix), data_ptr<half>(v_mix),
                   data_ptr<half>(r_mix), data_ptr<half>(kx),
                   data_ptr<half>(vx), data_ptr<half>(rx)},
               x.numel());

  int H = t_decay.size(0);
  int S = x.size(-1) / H;
  gemm_fp16_cublas_tensor(rx, rw, r);
  r = at::reshape(r, {H, 1, S});
  gemm_fp16_cublas_tensor(kx, kw, k);
  k = at::reshape(k, {H, S, 1});
  gemm_fp16_cublas_tensor(vx, vw, v);
  v = at::reshape(v, {H, 1, S});

  {
    Tensor a = at::matmul(k, v);

    // s1 = t_first * a + s
    // s2 = a + t_decay * s
    element_wise(Fused1{data_ptr<float>(t_first), data_ptr<float>(t_decay),
                        data_ptr<float>(a), data_ptr<float>(s),
                        static_cast<int32_t>(a.size(1) * a.size(2)),
                        data_ptr<float>(s1), data_ptr<float>(s2)},
                 a.numel());
  }

  Tensor out = at::matmul(r, s1);
  out = at::flatten(out);
  out = at::squeeze(at::group_norm(at::unsqueeze(out, 0), H, lx_w, lx_b), 0);
  out = at::_cast_Half(out);

  gemm_fp16_cublas_tensor(out, ow, x_plus_out);
  x_plus_out += x;
  return xx;
}
