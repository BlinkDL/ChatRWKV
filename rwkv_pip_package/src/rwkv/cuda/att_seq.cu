#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "util.h"
#include "element_wise.h"

using torch::Tensor;

void gemm_fp16_cublas(Tensor a, Tensor b, Tensor c);
void gemm_fp16_cublas(const void *a, const void *b, void *c, int m,
                      int n, int k, bool output_fp32);

// based on `kernel_wkv_forward`, fusing more operations
__global__ void kernel_wkv_forward_new(
    const int B, const int T, const int C, const float *__restrict__ const _w,
    const float *__restrict__ const _u, const float *__restrict__ const _k,
    const float *__restrict__ const _v, const half *__restrict__ const r,
    half *__restrict__ const _y, float *__restrict__ const _aa,
    float *__restrict__ const _bb, float *__restrict__ const _pp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;
  const int _state_offset = _b * C + _c;

  float u = _u[_c];
  float w = _w[_c];
  const float *__restrict__ const k = _k + _offset;
  const float *__restrict__ const v = _v + _offset;
  half *__restrict__ const y = _y + _offset;

  float aa = _aa[_state_offset];
  float bb = _bb[_state_offset];
  float pp = _pp[_state_offset];
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    const float kk = k[ii];
    const float vv = v[ii];
    float ww = u + kk;
    float p = max(pp, ww);
    float e1 = exp(pp - p);
    float e2 = exp(ww - p);
    y[ii] = __float2half((e1 * aa + e2 * vv) / (e1 * bb + e2));
    ww = w + pp;
    p = max(ww, kk);
    e1 = exp(ww - p);
    e2 = exp(kk - p);
    aa = e1 * aa + e2 * vv;
    bb = e1 * bb + e2;
    pp = p;
  }
  _aa[_state_offset] = aa;
  _bb[_state_offset] = bb;
  _pp[_state_offset] = pp;
}

void cuda_wkv_forward_new(int B, int T, int C, float *w, float *u, float *k,
                          float *v, half *r, half *y, float *aa, float *bb,
                          float *pp) {
  dim3 threadsPerBlock(min(C, 32));
  assert(B * C % threadsPerBlock.x == 0);
  dim3 numBlocks(B * C / threadsPerBlock.x);
  kernel_wkv_forward_new<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, r,
                                                         y, aa, bb, pp);
}

__global__ void _att_mix(const half *xx, const half *sx, const half *k_mix,
                        const half *v_mix, const half *r_mix,
                        const int outer_size, const int inner_size, half *kx,
                        half *vx, half *rx) {
  for (int idx2 = blockIdx.x * blockDim.x + threadIdx.x; idx2 < inner_size;
       idx2 += blockDim.x * gridDim.x) {
    half k_mix_ = k_mix[idx2];
    half v_mix_ = v_mix[idx2];
    half r_mix_ = r_mix[idx2];
    for (int row = 0; row < outer_size; ++row) {
      int idx1 = row * inner_size + idx2;
      half xx_ = xx[idx1];
      half sx_ = sx[idx1];
      kx[idx1] = __hadd(__hmul(xx_, k_mix_),
                        __hmul(sx_, __hsub(__float2half(1), k_mix_)));
      vx[idx1] = __hadd(__hmul(xx_, v_mix_),
                        __hmul(sx_, __hsub(__float2half(1), v_mix_)));
      rx[idx1] = __hadd(__hmul(xx_, r_mix_),
                        __hmul(sx_, __hsub(__float2half(1), r_mix_)));
    }
  }
}

void att_mix(const half *xx, const half *sx, const half *k_mix,
            const half *v_mix, const half *r_mix, const int outer_size,
            const int inner_size, half *kx, half *vx, half *rx) {
  // 256 is good enough on most GPUs
  const int32_t BLOCK_SIZE = 256;
  assert(inner_size % BLOCK_SIZE == 0);
  _att_mix<<<inner_size / BLOCK_SIZE, BLOCK_SIZE>>>(
      xx, sx, k_mix, v_mix, r_mix, outer_size, inner_size, kx, vx, rx);
}

struct InplaceSigmoid {
  __device__ __forceinline__ half operator()(int i) const {
    ptr[i] = __float2half(1.0 / (1.0 + exp(-__half2float(ptr[i]))));
  }
  half *ptr;
};

struct InplaceMul {
  __device__ __forceinline__ half operator()(int i) const {
    y[i] = __hmul(x[i], y[i]);
  }
  half *y;
  half *x;
};

/*
   Equivalent Python code:

   xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
   sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
   kx = xx * k_mix + sx * (1 - k_mix)
   vx = xx * v_mix + sx * (1 - v_mix)
   rx = xx * r_mix + sx * (1 - r_mix)

   r = torch.sigmoid(gemm(rx, rw))
   k = gemm(kx, kw, output_dtype=torch.float32)
   v = gemm(vx, vw, output_dtype=torch.float32)

   T = x.shape[0]
   for t in range(T):
       kk = k[t]
       vv = v[t]
       ww = t_first + kk
       p = torch.maximum(pp, ww)
       e1 = torch.exp(pp - p)
       e2 = torch.exp(ww - p)
       sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
       ww = t_decay + pp
       p = torch.maximum(ww, kk)
       e1 = torch.exp(ww - p)
       e2 = torch.exp(kk - p)
       aa = e1 * aa + e2 * vv
       bb = e1 * bb + e2
       pp = p
   out = gemm(r * sx, ow)
   return x + out, xx[-1,:], aa, bb, pp
*/
Tensor att_seq(Tensor x, Tensor sx, Tensor ln_w, Tensor ln_b, Tensor k_mix,
               Tensor v_mix, Tensor r_mix, Tensor kw, Tensor vw, Tensor rw,
               Tensor ow, Tensor t_first, Tensor pp, Tensor aa, Tensor bb,
               Tensor t_decay, /* imm */ Tensor buf, /* out */ Tensor x_plus_out) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  sx = at::cat({sx.unsqueeze(0), xx.slice(0, 0, -1)}, 0);
  char* buf_ptr = (char*)buf.data_ptr();
  half* kx = (half*)buf_ptr;
  half* vx = kx + x.numel();
  half* rx = vx + x.numel();
  half* wkv_y = rx + x.numel();
  att_mix(data_ptr<half>(xx), data_ptr<half>(sx), data_ptr<half>(k_mix),
         data_ptr<half>(v_mix), data_ptr<half>(r_mix), xx.size(0), xx.size(1),
         kx, vx, rx);
  float* k = reinterpret_cast<float*>(wkv_y + x.numel());
  float* v = k + x.size(0) * kw.size(1);
  half* r = reinterpret_cast<half*>(v + x.size(0) * vw.size(1));

  gemm_fp16_cublas(kx, kw.data_ptr(), k, x.size(0), kw.size(1), kw.size(0), true);
  gemm_fp16_cublas(vx, vw.data_ptr(), v, x.size(0), vw.size(1), vw.size(0), true);
  gemm_fp16_cublas(rx, rw.data_ptr(), r, x.size(0), rw.size(1), rw.size(0), false);
  element_wise(InplaceSigmoid{r}, x.size(0) * rw.size(1));
  cuda_wkv_forward_new(1, x.size(0), x.size(1), data_ptr<float>(t_decay),
                       data_ptr<float>(t_first), k, v, r,
                       wkv_y, data_ptr<float>(aa),
                       data_ptr<float>(bb), data_ptr<float>(pp));
  element_wise(InplaceMul{wkv_y, r}, x.numel());
  gemm_fp16_cublas(wkv_y, ow.data_ptr(), x_plus_out.data_ptr(), x.size(0), ow.size(1), ow.size(0), false);
  x_plus_out += x;
  return xx;
}
