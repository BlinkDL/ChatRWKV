#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace at {


template <typename scalar_t>
void kernel_impl(
    Tensor& xk,
    Tensor& xr,
    const Tensor& input,
    const Tensor& state,
    int64_t index,
    const Tensor& k,
    const Tensor& r) {

  int64_t K = state.size(1);

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* k_data = k.data_ptr<scalar_t>();
  scalar_t* r_data = r.data_ptr<scalar_t>();
  scalar_t* xk_data = xk.data_ptr<scalar_t>();
  scalar_t* xr_data = xr.data_ptr<scalar_t>();

  // add offset
  scalar_t* state_data = state.data_ptr<scalar_t>();
  scalar_t* s_data = state_data + 5 * index * K;

  #pragma omp simd
  for (int64_t i = 0; i < K; ++i) {
    xk_data[i] = input_data[i] * k_data[i] + s_data[i] * (1 - k_data[i]);
    xr_data[i] = input_data[i] * r_data[i] + s_data[i] * (1 - r_data[i]);
    s_data[i] = input_data[i];
  }
}

Tensor channel_mixing_kernel(
    const Tensor& input,
    const Tensor& state,
    int64_t index,
    const Tensor& k,
    const Tensor& r,
    const Tensor& kw,
    const Tensor& vw,
    const Tensor& rw) {

  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

  int64_t K = input.size(0);
  auto xk = at::zeros({K}, input.options());
  auto xr = at::zeros({K}, input.options());
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_mixing_kernel", [&]() {
    return kernel_impl<scalar_t>(xk, xr, input, state, index, k, r);
  });

  auto rr = torch::sigmoid(torch::matmul(rw, xr));
  auto kk = torch::square(torch::relu(torch::matmul(kw, xk)));
  return rr * torch::matmul(vw, kk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("channel_mixing", &channel_mixing_kernel, "RWKV channel mixing forward");
}

} // namespace at
