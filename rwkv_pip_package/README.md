The RWKV Language Model

https://github.com/BlinkDL/ChatRWKV

https://github.com/BlinkDL/RWKV-LM

```python
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' #  if '1' then compile CUDA kernel for seq mode (much faster)

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
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
# ########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

pipeline = PIPELINE(model, "20B_tokenizer.json") # find it in https://github.com/BlinkDL/ChatRWKV

# download models: https://huggingface.co/BlinkDL
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32')

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7,
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = []) # stop generation whenever you see any token here

pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
print('\n')

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
```