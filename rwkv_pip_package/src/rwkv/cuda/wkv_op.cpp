#include <torch/extension.h>

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *aa, float *bb, float *pp);

void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
}
