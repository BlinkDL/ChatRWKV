#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#define MIN_VALUE (-1e38)
typedef at::Half fp16;

__global__ void kernel_wkv_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w, const float *__restrict__ const _u, const fp16 *__restrict__ const _k, const fp16 *__restrict__ const _v,
                               fp16 *__restrict__ const _y, float *__restrict__ const _aa, float *__restrict__ const _bb, float *__restrict__ const _pp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _state_offset = _b * C + _c;

    float u = _u[_c];
    float w = _w[_c];
    const fp16 *__restrict__ const k = _k + _offset;
    const fp16 *__restrict__ const v = _v + _offset;
    fp16 *__restrict__ const y = _y + _offset;

    float aa = _aa[_state_offset];
    float bb = _bb[_state_offset];
    float pp = _pp[_state_offset];
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = fp16((e1 * aa + e2 * vv) / (e1 * bb + e2));
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

void cuda_wkv_forward(int B, int T, int C, float *w, float *u, fp16 *k, fp16 *v, fp16 *y, float *aa, float *bb, float *pp) {
    dim3 threadsPerBlock( min(C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_wkv_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, aa, bb, pp);
}

__half *cast(fp16 *ptr) {
    return reinterpret_cast<__half *>(ptr);
}

__global__ void kernel_mm8_seq(
    const int B, const int N, const int M,
    const __half *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    __half *__restrict__ const y, const int y_stride) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M) {
        float y_local = 0;
        for (int j = 0; j < N; ++j) {
            y_local += __half2float(x[i * x_stride + j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        y[i * y_stride + k] = __float2half(y_local);
    }
}
void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  fp16 *y, int y_stride) {
    dim3 blockSize(1, 128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_seq<<<gridSize, blockSize>>>(
        B, N, M, cast(x), x_stride, w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), cast(y), y_stride);
}


#if defined OPTIMIZED_MM8

#define MM8_CHUNK     16    // Needs to be power of 2 and <= 32 (warp size)
#define MM8_THREADS   64    // Needs to be multiple of 32
#define MM8_BLOCKS  1024

__global__ void kernel_mm8_one(
    const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {
    
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    float4  sum = {0.0f, 0.0f, 0.0f, 0.0f};
    float   sum_x = 0;
    float   sum_x_my = 0;

    for (int i = MM8_CHUNK * blockIdx.y; i < N; i += MM8_CHUNK * gridDim.y) {
        if (j < M) {
#pragma unroll
            for (int k = 0; k < MM8_CHUNK; ++k) {
                float   x_ry_i  = __half2float(x[i + k]) * __half2float(ry[i + k]);
                uchar4  w_ij    = *(uchar4*)(w + (i + k) * w_stride + j);   // Read 4 at once for efficiency

                sum.x += x_ry_i * (float(w_ij.x) + 0.5f);
                sum.y += x_ry_i * (float(w_ij.y) + 0.5f);
                sum.z += x_ry_i * (float(w_ij.z) + 0.5f);
                sum.w += x_ry_i * (float(w_ij.w) + 0.5f);
            }
        }

        int k = threadIdx.x % MM8_CHUNK;
        float x_i   = __half2float(x[i + k]);
        float my_i  = __half2float(my[i + k]);
        sum_x       += x_i;
        sum_x_my    += x_i * my_i;
    }

    // Each thread only summed a subset, reduce so everyone has the full sum
    for (int offset = MM8_CHUNK / 2; offset > 0; offset /= 2) {
        sum_x       += __shfl_xor_sync(0xffffffff, sum_x, offset, MM8_CHUNK);
        sum_x_my    += __shfl_xor_sync(0xffffffff, sum_x_my, offset, MM8_CHUNK);
    }

    // We might need some threads with j >= M to calculate sum_x and sum_mx_x correctly, so we can't move this earlier
    if (j >= M) return;

    float2 rx_j_0   = __half22float2(*(half2*)(rx + j));
    float2 rx_j_1   = __half22float2(*(half2*)(rx + j + 2));
    float2 mx_j_0   = __half22float2(*(half2*)(mx + j));
    float2 mx_j_1   = __half22float2(*(half2*)(mx + j + 2));
    atomicAdd(&y[j + 0], sum.x * rx_j_0.x + sum_x * mx_j_0.x + sum_x_my);
    atomicAdd(&y[j + 1], sum.y * rx_j_0.y + sum_x * mx_j_0.y + sum_x_my);
    atomicAdd(&y[j + 2], sum.z * rx_j_1.x + sum_x * mx_j_1.x + sum_x_my);
    atomicAdd(&y[j + 3], sum.w * rx_j_1.y + sum_x * mx_j_1.y + sum_x_my);
}

void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  float *y) {
    assert(N % MM8_CHUNK == 0);
    assert(M % 4 == 0);
    assert(w_stride % 4 == 0);

    int num_blocks_x = (M - 1) / (MM8_THREADS * 4) + 1;
    dim3 blockSize(MM8_THREADS);
    dim3 gridSize(num_blocks_x, min(MM8_BLOCKS / num_blocks_x, N / MM8_CHUNK));

    kernel_mm8_one<<<gridSize, blockSize>>>(
        N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y);
}

#else

#define MM8_ONE_JSPLIT 24
#define MM8_ONE_TILE 1024

__global__ void kernel_mm8_one(
    const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += __half2float(x[j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        atomicAdd(&y[k], y_local);
    }
}
void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  float *y) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_one<<<gridSize, blockSize>>>(
        N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y);
}
#endif
