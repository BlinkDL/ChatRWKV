#include "util.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// NOTE: float4 is a overkill for current sizes (4096 in 7B model and 768 in 0.1B model),
// and is not faster than the plain float version.
// Now the plain float version is used.
__global__ void wkv_forward_one(float4 *t_first, float4 *k, float4 *pp,
                                float4 *aa, float4 *bb, float4 *t_decay,
                                float4 *v, /* out */ float4 *t1,
                                /* out */ float4 *t2, /* out */ float4 *p,
                                /* in & out, half */ float2 *r,
                                unsigned int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4;
       i += blockDim.x * gridDim.x) {
    float4 ww = t_first[i] + k[i];
    float4 pp_ = pp[i];
    float4 p_ = fmaxf(pp_, ww);
    float4 e1 = expf(pp_ - p_);
    float4 e2 = expf(ww - p_);

    float4 aa_ = aa[i];
    float4 bb_ = bb[i];
    float4 v_ = v[i];
    half2 wkv1 = make_half2(
        __float2half(((e1.x * aa_.x + e2.x * v_.x) / (e1.x * bb_.x + e2.x))),
        __float2half(((e1.y * aa_.y + e2.y * v_.y) / (e1.y * bb_.y + e2.y))));
    half2 wkv2 = make_half2(
        __float2half(((e1.z * aa_.z + e2.z * v_.z) / (e1.z * bb_.z + e2.z))),
        __float2half(((e1.w * aa_.w + e2.w * v_.w) / (e1.w * bb_.w + e2.w))));
    half2 *r1 = reinterpret_cast<half2 *>(&r[i].x);
    half2 *r2 = reinterpret_cast<half2 *>(&r[i].y);
    *r1 = __hmul2(wkv1, *r1);
    *r2 = __hmul2(wkv2, *r2);

    ww = t_decay[i] + pp_;
    float4 k_ = k[i];
    p_ = fmaxf(ww, k_);
    e1 = expf(ww - p_);
    e2 = expf(k_ - p_);

    t1[i] = e1 * aa_ + e2 * v_;
    t2[i] = e1 * bb_ + e2;

    p[i] = p_;
  }
}

__global__ void wkv_forward_one(float *t_first, float *k, float *pp, float *aa,
                                float *bb, float *t_decay, float *v,
                                /* out */ float *t1, /* out */ float *t2,
                                /* out */ float *p, /* in & out */ half *r,
                                unsigned int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
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
}
