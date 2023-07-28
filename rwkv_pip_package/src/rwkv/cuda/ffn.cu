#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "element_wise.h"
#include "util.h"

using torch::Tensor;

void gemm_fp16_cublas(const void *a, const void *b, void *c, int ori_m,
                      int ori_n, int ori_k, bool output_fp32);

__global__ void _ffn_seq_mix(const half *xx, const half *sx, const half *k_mix,
                             const half *r_mix, const int outer_size,
                             const int inner_size, half *kx, half *rx) {
  for (int idx2 = blockIdx.x * blockDim.x + threadIdx.x; idx2 < inner_size;
       idx2 += blockDim.x * gridDim.x) {
    half k_mix_ = k_mix[idx2];
    half r_mix_ = r_mix[idx2];
    for (int row = 0; row < outer_size; ++row) {
      int idx1 = row * inner_size + idx2;
      half xx_ = xx[idx1];
      half sx_ = sx[idx1];
      kx[idx1] = __hadd(__hmul(xx_, k_mix_),
                        __hmul(sx_, __hsub(__float2half(1), k_mix_)));
      rx[idx1] = __hadd(__hmul(xx_, r_mix_),
                        __hmul(sx_, __hsub(__float2half(1), r_mix_)));
    }
  }
}

void ffn_seq_mix(const half *xx, const half *sx, const half *k_mix,
                 const half *r_mix, const int outer_size, const int inner_size,
                 half *kx, half *rx) {
  // 256 is good enough on most GPUs
  const int32_t BLOCK_SIZE = 256;
  assert(inner_size % BLOCK_SIZE == 0);
  _ffn_seq_mix<<<inner_size / BLOCK_SIZE, BLOCK_SIZE>>>(
      xx, sx, k_mix, r_mix, outer_size, inner_size, kx, rx);
}

struct InplaceSigmoid {
  __device__ __forceinline__ void operator()(int i) const {
    ptr[i] = __float2half(1.0 / (1.0 + exp(-__half2float(ptr[i]))));
  }
  half *ptr;
};

struct InplaceReLUAndSquare {
  __device__ __forceinline__ void operator()(int i) const {
    // __hmax is not defined in old cuda
    if (__hgt(ptr[i], __float2half(0))) {
      ptr[i] = __hmul(ptr[i], ptr[i]);
    } else {
      ptr[i] = __float2half(0);
    }
  }
  half *ptr;
};

struct InplaceFma {
  __device__ __forceinline__ void operator()(int i) const {
    a[i] = __hfma(a[i], b[i], c[i]);
  }
  half *a;
  const half *b;
  const half *c;
};

/*
   Equivalent Python code:

   xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
   sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
   kx = xx * k_mix + sx * (1 - k_mix)
   rx = xx * r_mix + sx * (1 - r_mix)

   r = torch.sigmoid(gemm(rx, rw))
   vx = torch.square(torch.relu(gemm(kx, kw)))
   out = r * gemm(vx, vw)
   return x + out, xx[-1,:]
*/
Tensor ffn_seq(Tensor x, Tensor sx, Tensor ln_w, Tensor ln_b, Tensor k_mix,
               Tensor r_mix, Tensor kw, Tensor vw, Tensor rw,
               /* imm */ Tensor buf,
               /* out */ Tensor x_plus_out) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  sx = at::cat({sx.unsqueeze(0), xx.slice(0, 0, -1)}, 0);
  char *buf_ptr = (char *)buf.data_ptr();
  half *kx = (half *)buf_ptr;
  half *rx = kx + x.numel();
  half *vx = rx + x.numel();
  half *r = vx + x.size(0) * kw.size(1);
  ffn_seq_mix(data_ptr<half>(xx), data_ptr<half>(sx), data_ptr<half>(k_mix),
              data_ptr<half>(r_mix), xx.size(0), xx.size(1), kx, rx);

  gemm_fp16_cublas(rx, rw.data_ptr(), r, x.size(0), rw.size(1), x.size(1),
                   false);
  element_wise(InplaceSigmoid{r}, x.size(0) * rw.size(1));
  gemm_fp16_cublas(kx, kw.data_ptr(), vx, x.size(0), kw.size(1), x.size(1),
                   false);
  element_wise(InplaceReLUAndSquare{vx}, x.size(0) * kw.size(1));
  gemm_fp16_cublas(vx, vw.data_ptr(), x_plus_out.data_ptr(), x.size(0),
                   vw.size(1), vw.size(0), false);
  element_wise(InplaceFma{data_ptr<half>(x_plus_out), r, data_ptr<half>(x)},
               x_plus_out.numel());
  return xx;
}

struct FfnOneMix {
  __device__ __forceinline__ void operator()(int idx) {
    half k_mix_ = k_mix[idx];
    half r_mix_ = r_mix[idx];
    half xx_ = xx[idx];
    half sx_ = sx[idx];
    kx[idx] = __hadd(__hmul(xx_, k_mix_),
                     __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    rx[idx] = __hadd(__hmul(xx_, r_mix_),
                     __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
  half *k_mix;
  half *r_mix;
  half *xx;
  half *sx;
  half *kx;
  half *rx;
};

/*
  Equivalent Python code:

  xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
  kx = xx * k_mix + sx * (1 - k_mix)
  rx = xx * r_mix + sx * (1 - r_mix)

  r = torch.sigmoid(gemm(rx, rw))
  vx = torch.square(torch.relu(gemm(kx, kw)))
  out = r * gemm(vx, vw)
  return x + out, xx
*/
Tensor ffn_one(Tensor x, Tensor sx, Tensor ln_w, Tensor ln_b, Tensor k_mix,
               Tensor r_mix, Tensor kw, Tensor vw, Tensor rw,
               /* imm */ Tensor buf,
               /* out */ Tensor x_plus_out) {
  Tensor xx = at::layer_norm(x, {x.size(-1)}, ln_w, ln_b);
  char *buf_ptr = (char *)buf.data_ptr();
  half *kx = (half *)buf_ptr;
  half *rx = kx + x.numel();
  half *vx = rx + x.numel();
  half *r = vx + x.size(0) * kw.size(1);
  element_wise(FfnOneMix{data_ptr<half>(k_mix), data_ptr<half>(r_mix),
                         data_ptr<half>(xx), data_ptr<half>(sx), kx, rx},
               x.numel());
  // vector * matrix, so m = 1
  gemm_fp16_cublas(rx, rw.data_ptr(), r, 1, rw.size(1), rw.size(0), false);
  element_wise(InplaceSigmoid{r}, rw.size(1));
  gemm_fp16_cublas(kx, kw.data_ptr(), vx, 1, kw.size(1), kw.size(0), false);
  element_wise(InplaceReLUAndSquare{vx}, kw.size(1));
  gemm_fp16_cublas(vx, vw.data_ptr(), x_plus_out.data_ptr(), 1, vw.size(1),
                   vw.size(0), false);
  element_wise(InplaceFma{data_ptr<half>(x_plus_out), r, data_ptr<half>(x)},
               x_plus_out.numel());
  return xx;
}
