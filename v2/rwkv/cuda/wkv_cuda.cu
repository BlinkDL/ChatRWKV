#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward( const int B, const int T, const int C,
                                const F *__restrict__ const _w,
                                const F *__restrict__ const _u,
                                const F *__restrict__ const _k,
                                const F *__restrict__ const _v,
                               F *__restrict__ const _y,
                               F *__restrict__ const _aa,
                               F *__restrict__ const _bb,
                               F *__restrict__ const _pp) {

    // The input is a 3-dimension matrix has shape of (B, T, C)
    // Each cuda thread process a single channel of (B, C)
    // There is B * C total channels of B samples
    // `idx = _b * C + _c` point to (_b, _c) element of (B, C) matrix
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C; // _b: current sample of the batch
    const int _c = idx % C; // _c: current channel

    // Each sample _b has T tokens represented by (T, C) maxtrix of type F
    const int _offset = _b * T * C + _c; // point to scalar values of current channel
    const int _state_offset = _b * C + _c; // point to scalar values of model's current states

    // Follow RWKV4 RNN formular https://github.com/BlinkDL/RWKV-LM#rwkv-4-improvements, we have:
    F u = _u[_c]; // u of current channel
    F w = _w[_c]; // w of current channel

    const F *__restrict__ const k = _k + _offset; // point to k values of current channel
    const F *__restrict__ const v = _v + _offset; // point to v values of current channel
    F *__restrict__ const y = _y + _offset; //  point to output values of current channel

    // aa, bb, pp is model's current states
    F aa = _aa[_state_offset];
    F bb = _bb[_state_offset];
    F pp = _pp[_state_offset];

    // With each token in T input tokens
    for (int i = 0; i < T; i++) {
        const int ii = i * C; // index of current channel of i-th token
        F kk = k[ii]; // key of current channel of i-th token
        F vv = v[ii]; // value of current channel of i-th token
        F ww = u + kk;
        F p = max(pp, ww); // this is `q` in RNN fomular (new)
        F e1 = exp(pp - p);
        F e2 = exp(ww - p);
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2); // this is `c / d` in RNN fomular (new)
        p = max(w + pp, kk); // this is `q` in RNN fomular (new)
        e1 = exp(w + pp - p);
        e2 = exp(kk - p);
        // Calculate current aa, bb, pp
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    // Update model states with new aa, bb, pp values
    _aa[_state_offset] = aa;
    _bb[_state_offset] = bb;
    _pp[_state_offset] = pp;
}

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, 
                    float *aa, float *bb, float *pp) {
    dim3 threadsPerBlock( min(C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, aa, bb, pp);
}
