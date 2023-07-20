#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "element_wise.h"
#include "util.h"

// Equivalent Python code:
// ww = t_first + k
// p = torch.maximum(pp, ww)
// e1 = torch.exp(pp - p)
// e2 = torch.exp(ww - p)
// wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
// ww = t_decay + pp
// p = torch.maximum(ww, k)
// e1 = torch.exp(ww - p)
// e2 = torch.exp(k - p)
// t1 = e1 * aa + e2 * v
// t2 = e1 * bb + e2
// r = r * wkv
// return t1, t2, p, r
struct WkvForwardOne {
  const float *t_first;
  const float *k;
  const float *pp;
  const float *aa;
  const float *bb;
  const float *t_decay;
  const float *v;
  /* out */ float *t1;
  /* out */ float *t2;
  /* out */ float *p;
  /* in & out */ half *r;

  __device__ void operator()(int i) const {
    float ww = t_first[i] + k[i];
    float pp_ = pp[i];
    float p_ = (pp_ > ww) ? pp_ : ww;
    float e1 = expf(pp_ - p_);
    float e2 = expf(ww - p_);
    float aa_ = aa[i];
    float bb_ = bb[i];
    float v_ = v[i];
    r[i] = __hmul(r[i], __float2half(((e1 * aa_ + e2 * v_) / (e1 * bb_ + e2))));
    ww = t_decay[i] + pp_;
    float k_ = k[i];
    p_ = (ww > k_) ? ww : k_;
    e1 = expf(ww - p_);
    e2 = expf(k_ - p_);
    t1[i] = e1 * aa_ + e2 * v_;
    t2[i] = e1 * bb_ + e2;
    p[i] = p_;
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

void gemm_fp16_cublas(Tensor a, Tensor b, Tensor c);

Tensor att_one(Tensor x, Tensor ln_w, Tensor ln_b, Tensor sx, Tensor k_mix,
             Tensor v_mix, Tensor r_mix, Tensor kw,
             /* imm */ Tensor kx, Tensor vw, /* imm */ Tensor vx, Tensor rw,
             /* imm */ Tensor rx, Tensor ow, Tensor t_first,
             /* imm */ Tensor k, Tensor pp, Tensor ww, Tensor aa, Tensor bb,
             Tensor t_decay, /* imm */ Tensor v, /* in & out */ Tensor r,
             /* out */ Tensor x_plus_out, /* out */ Tensor t1,
             /* out */ Tensor t2, /* out */ Tensor p) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  element_wise(Mix{data_ptr<half>(xx), data_ptr<half>(sx),
                   data_ptr<half>(k_mix), data_ptr<half>(v_mix),
                   data_ptr<half>(r_mix), data_ptr<half>(kx),
                   data_ptr<half>(vx), data_ptr<half>(rx)},
               x.numel());

  gemm_fp16_cublas(kx, kw, k);
  gemm_fp16_cublas(vx, vw, v);
  gemm_fp16_cublas(rx, rw, r);
  at::sigmoid_(r);

  element_wise(WkvForwardOne{data_ptr<float>(t_first), data_ptr<float>(k),
                             data_ptr<float>(pp), data_ptr<float>(aa),
                             data_ptr<float>(bb), data_ptr<float>(t_decay),
                             data_ptr<float>(v), data_ptr<float>(t1),
                             data_ptr<float>(t2), data_ptr<float>(p),
                             data_ptr<half>(r)},
               x.numel());

  gemm_fp16_cublas(r, ow, x_plus_out);
  x_plus_out += x;
  return xx;
}
