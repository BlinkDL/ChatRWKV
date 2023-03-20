import torch, sys
from time import perf_counter as time
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

torch.manual_seed(0)

use_new = 1

from torch.utils.cpp_extension import load
current_path = '../rwkv_pip_package/src/rwkv/'
load(
    name=f"wkv_cuda",
    sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
    verbose=True,
    extra_cuda_cflags=["-t 4", "-std=c++17", "--use_fast_math", "-O3", "--extra-device-vectorization"]+["-DOPTIMIZED_MM8"]*use_new,
    is_python_module=False)

@torch.jit.script
def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
    assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype == torch.float16
    assert w.dtype == torch.uint8
    assert x.shape == [N]
    assert w.shape == [N, M]
    assert rx.shape == mx.shape == [M]
    assert ry.shape == my.shape == [N, 1]
    y = torch.zeros((M,), device='cuda', dtype=torch.float32)
    torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
    return y.to(dtype=torch.float16)

def mm8_one(x, w, mx, rx, my, ry):
    N, M = w.shape[0], w.shape[1]
    return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)

def mm8_one_truth(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

N_list = list(range(16,2000,16))+[2560,5120,5120*2]
M_list = list(range(4,2000,4))+[2560,5120,5120*2]
max_error = 0
for N in tqdm(N_list):
  for M in M_list:
    l = []
    for dim,dtype in [([N], 5), ([N, M], 0), ([M], 5), ([M], 5), ([N, 1], 5), ([N, 1], 5)]:
      if dtype == 5:
        l.append(torch.randn(tuple(dim),device='cuda',dtype=torch.float16))
      else:
        l.append(torch.randint(low=0,high=255,size=tuple(dim),device='cuda',dtype=torch.uint8))

    x,w,mx,rx,my,ry = l
    y0 = mm8_one(x, w, mx, rx, my, ry)
    y1 = mm8_one_truth(x, w, mx, rx, my, ry)

    err = ((y1-y0).norm()/y0.norm()).item()
    max_error = max(max_error, err)
print(max_error)
