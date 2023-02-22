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
# use '/' in model path, instead of '\'
#
# can split the model to two devices (cpu/cuda/cuda:0/cuda:1/...) and set dtype for each of them
# the first [dev1_layers] layers goes to [dev1]
#
# fp16 - good for GPU, DOES NOT support CPU
# fp32 - good for CPU
# bf16 - worse accuracy, supports CPU
#
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', dev1='cpu', dtype1='fp32', dev2='cuda', dtype2='fp16', dev1_layers=6)

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

# input('done')
