########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('\nChatRWKV v2 (!!! WIP, might be buggy !!!) https://github.com/BlinkDL/ChatRWKV\n')

import os, time, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)
os.environ["RWKV_JIT_ON"] = '1'

from rwkv.model import RWKV
from rwkv.utils import TOKENIZER
tokenizer = TOKENIZER("20B_tokenizer.json")

########################################################################################################
#
# Use '/' in model path, instead of '\'
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
#
# Strategy examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# Here we consider [ln_out+head] to be an extra layer, so L12-D768 model has "13" layers, L24-D2048 model has "25" layers, etc.
#
# 'cpu fp32' = everything on cpu fp32
# 'cuda fp16' = everything on cuda fp16
#
# 'cuda fp16 *6 -> cpu fp32' = first 6 layers on cuda fp16, then on cpu fp32
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers on cuda:0 fp16, then 8 layers on cuda:1 fp16, then on cpu fp32
#
# Use '+' for STREAM mode (do it on your fastest GPU), requires some VRAM to store streamed layers
# 'cuda fp16 *6+' = first 6 layers on cuda fp16, then stream the rest on it
# (for best speed: try *1+ *2+ *3+ ... until you run out of VRAM)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16 *0+ -> cpu fp32 *1' = stream all layers on cuda fp16, then [ln_out+head] on cpu fp32
#
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cpu fp32')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *8 -> cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda:0 fp16 -> cuda:1 fp16 -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *6+')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019', strategy='cuda fp16 *0+ -> cpu fp32 *1')

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())
out, state = model.forward([187], None)
print(out.detach().cpu().numpy())
out, state = model.forward([510, 1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187], None)
out, state = model.forward([510, 1563, 310, 247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())

input('done. press Ctrl+C to exit')
