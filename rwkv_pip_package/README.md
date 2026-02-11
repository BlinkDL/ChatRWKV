The RWKV Language Model

https://rwkv.com


```python
#
# !!! set these os.environ[] before import RWKV !!!
#
import os
os.environ["RWKV_V7_ON"] = '1' # !!! enable RWKV-7 !!!
# os.environ["RWKV_DE_VERSION"] = '1' # enable DeepEmbed if applicable
os.environ['RWKV_JIT_ON'] = '1' # '1' for better speed
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster prefilling), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
#
# download models: https://huggingface.co/BlinkDL
# try strategy='cuda fp16' or 'cpu fp32'
#
model = RWKV(model='/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096', strategy='cuda fp16') # Use '/' in model path, instead of '\'

pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # for "g" and "world" models
# pipeline = PIPELINE(model, "20B_tokenizer.json") # for "pile" models, 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

ctx = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
print(ctx, end='')

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.5, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     alpha_decay = 0.997, # gradually decay the penalty
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

def my_print(s):
    print(s, end='', flush=True)

pipeline.generate(ctx, token_count=500, args=args, callback=my_print)
print('\n')

# !!! model.forward(tokens, state) will modify state in-place !!!

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
```

Faster decoding (CUDAGraph, requires rwkv pip pkg v0.8.31+):
```python
import os, time
import numpy as np
import torch
os.environ["RWKV_V7_ON"] = '1'
# os.environ["RWKV_DE_VERSION"] = '1' # enable DeepEmbed if applicable
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' 
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
model = RWKV(model='/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096', strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"

###############################################################################

print('='*80 + '\nSlow inference\n' + '='*80)
print(prompt, end="")

all_tokens = []
out_last = 0
out, state = model.forward(pipeline.encode(prompt), None)

times = []
all_times = []
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = pipeline.sample_logits(out, temperature=TEMPERATURE, top_p=TOP_P)
    all_tokens += [token]

    tmp = pipeline.decode(all_tokens[out_last:])
    if '\ufffd' not in tmp:
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1    

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out, state = model.forward(token, state)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
times = np.percentile(times, 50)
all_times = np.percentile(all_times, 50)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full)')

###############################################################################

print('='*80 + '\nFast inference (CUDAGraph, requires rwkv pip pkg v0.8.31+)\n' + '='*80)
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state()

static_input = torch.empty((model.n_embd), device="cuda", dtype=torch.half)
static_state_in = [torch.empty_like(x, device="cuda") for x in state]
static_state_out = [torch.empty_like(x, device="cuda") for x in state]
static_output = torch.empty((model.args.vocab_size), device="cuda", dtype=torch.half)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output, static_state_out = model.forward_one_alt(static_input, static_state_in)

out, state = model.forward(pipeline.encode(prompt), state)
for i in range(len(state)):
    static_state_in[i].copy_(state[i])
static_output.copy_(out)

times = []
all_times = []
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = pipeline.sample_logits(static_output, temperature=TEMPERATURE, top_p=TOP_P)
    all_tokens += [token]

    tmp = pipeline.decode(all_tokens[out_last:])
    if '\ufffd' not in tmp:
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    static_input.copy_(model.z['emb.weight'][token])
    g.replay()
    for n in range(len(state)):
        static_state_in[n].copy_(static_state_out[n])
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
times = np.percentile(times, 50)
all_times = np.percentile(all_times, 50)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) (note: very inefficient sample_logits)')
```

Old readme:

```python
########################################################################################################
#
# For RWKV-4/5/6 models:
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU
# fp32 = good for CPU
# bf16 = supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################
```
