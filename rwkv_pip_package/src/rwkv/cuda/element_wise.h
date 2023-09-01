#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename Func> __global__ void _element_wise(Func func, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    func(i);
  }
}

// NOTE: packed data type (e.g. float4) is a overkill for current sizes
// (4096 in 7B model and 768 in 0.1B model),
// and is not faster than the plain float version.
template <typename Func>
void element_wise(Func func, int n) {
  // 256 is good enough on most GPUs
  const int32_t BLOCK_SIZE = 256;
  assert(n % BLOCK_SIZE == 0);
  _element_wise<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(func, n);
}
