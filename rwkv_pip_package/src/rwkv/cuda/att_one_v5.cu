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

using torch::Tensor;

void gemm_cublas_tensor(const Tensor &a, const Tensor &b, const Tensor &c);

// xx = torch.ops.rwkv.att_one_v5(x, sx, s, ln_w, ln_b, lx_w, lx_b, kvr_mix,
// kvrx, kvrw, ow, t_first, t_decay, kvr, a, buf, s1, x_plus_out, s2) # type:
// ignore[reportGeneralTypeIssues]
Tensor att_one_v5(Tensor x, Tensor sx, Tensor s, Tensor ln_w, Tensor ln_b,
                  Tensor lx_w, Tensor lx_b, Tensor kvr_mix,
                  /* imm */ Tensor kvrx, Tensor kvrw, Tensor ow, Tensor t_first,
                  Tensor t_decay, /* imm */ Tensor kvr, /* imm */ Tensor a,
                  /* imm */ Tensor buf,
                  /* imm */ Tensor s1,
                  /* out */ Tensor x_plus_out, /* out */ Tensor s2) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  element_wise(Mix{data_ptr<half>(xx), data_ptr<half>(sx),
                   data_ptr<half>(kvr_mix),
                   static_cast<int>(x.numel()), data_ptr<half>(kvrx)},
               x.numel());

  int H = t_decay.size(0);
  int S = x.size(-1) / H;
  gemm_cublas_tensor(at::unsqueeze(kvrx, 1), kvrw, kvr);
  Tensor k = at::reshape(kvr[0], {H, S, 1});
  Tensor v = at::reshape(kvr[1], {H, 1, S});
  Tensor r = at::reshape(kvr[2], {H, 1, S});

  gemm_cublas_tensor(k, v, a);
  // s1 = t_first * a + s
  // s2 = a + t_decay * s
  element_wise(Fused1{data_ptr<float>(t_first), data_ptr<float>(t_decay),
                      data_ptr<float>(a), data_ptr<float>(s),
                      static_cast<int32_t>(a.size(1) * a.size(2)),
                      data_ptr<float>(s1), data_ptr<float>(s2)},
               a.numel());

  gemm_cublas_tensor(r, s1, buf);
  buf = at::flatten(buf);
  buf = at::squeeze(at::group_norm(at::unsqueeze(buf, 0), H, lx_w, lx_b), 0);
  buf = at::_cast_Half(buf);

  gemm_cublas_tensor(buf, ow, x_plus_out);
  x_plus_out += x;
  return xx;
}
