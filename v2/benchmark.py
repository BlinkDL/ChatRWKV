########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

########################################################################################################

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230204-7324'
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = [187]

########################################################################################################

print(f'\nLoading ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
pipeline = PIPELINE(model, "20B_tokenizer.json")

print('Warmup...')
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
out, state = model.forward([187, 510, 1563, 310], None)
out, state = model.forward([247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())

########################################################################################################

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    logits = 0
    correct = True
    for i in range(len(dst)):
        if i == 0:
            out, model_state = model.forward(src, None)
        else:
            out, model_state = model.forward([dst[i-1]], model_state)
        probs = F.softmax(out.float(), dim=-1)
        logits += math.log(probs[dst[i]])
        _, s_index = torch.sort(probs, descending=True)
        pred = s_index[0].item()
        if pred != dst[i]:
            correct = False
    
    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))#, 'pred', pred, 'dst', dst)
