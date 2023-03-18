########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('\nChatRWKV https://github.com/BlinkDL/ChatRWKV\n')

import os, sys, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(f'{current_path}/rwkv_pip_package/src')

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# Read https://pypi.org/project/rwkv/ for Strategy Guide
#
########################################################################################################
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV # pip install rwkv
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8 *6 -> cuda fp16 *0+ -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32 *3 -> cuda fp16 *6+')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *8 -> cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda:0 fp16 -> cuda:1 fp16 -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *6+')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019', strategy='cuda fp16 *0+ -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096', strategy='cuda:0 fp16 *25 -> cuda:1 fp16')

# out, state = model.forward([187], None)
# print(out.detach().cpu().numpy())

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above

# print('\n')
# exit(0)
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################
# 1. set os.environ["RWKV_CUDA_ON"] = '1' if possible, for faster preprocess of a long ctx.
# 2. Reuse the state (use deepcopy to clone it) when you are running the same ctx multiple times. 
pipeline.generate(ctx, token_count=200, args=args, callback=my_print)

print('\n')
