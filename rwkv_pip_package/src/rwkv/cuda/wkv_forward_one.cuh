#include "cuda_runtime.h"

__global__ void wkv_forward_one(float4 *t_first, float4 *k, float4 *pp,
                                float4 *aa, float4 *bb, float4 *t_decay,
                                float4 *v, /* out */ float4 *t1,
                                /* out */ float4 *t2, /* out */ float4 *p,
                                /* in & out, half */ float2 *r, unsigned int n);
__global__ void wkv_forward_one(float *t_first, float *k, float *pp, float *aa,
                                float *bb, float *t_decay, float *v,
                                /* out */ float *t1, /* out */ float *t2,
                                /* out */ float *p, /* in & out */ half *r,
                                unsigned int n);
