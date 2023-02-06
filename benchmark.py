########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]
args = types.SimpleNamespace()

########################################################################################################

args.RUN_DEVICE = "cuda" # cuda / cpu 
args.FLOAT_MODE = "fp16" # fp16 / fp32 / bf16
os.environ["RWKV_JIT_ON"] = '1' # 0 / 1
args.ctx_len = 1024

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230128-6782'
# args.n_layer = 40
# args.n_embd = 5120

args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
args.n_layer = 24
args.n_embd = 2048

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'
# args.n_layer = 12
# args.n_embd = 768

PAD_SEQ = [187]

########################################################################################################

print(f'\nLoading ChatRWKV - {args.RUN_DEVICE} - {args.FLOAT_MODE}')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
from src.model_run import RWKV_RNN
from src.utils import TOKENIZER
tokenizer = TOKENIZER("20B_tokenizer.json")

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

MODEL_NAME = args.MODEL_NAME
print(f'Loading model - {MODEL_NAME}')
model = RWKV_RNN(args)

print('Running...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = PAD_SEQ + tokenizer.encode(d[0])
    dst = tokenizer.encode(d[1])

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
