########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time
from typing import List
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

########################################################################################################

@torch.jit.script
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
            # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

########################################################################################################

# cuda bf16 inference

tokenizer = RWKV_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

args = types.SimpleNamespace()
args.MODEL_NAME = '/home/rwkv/rwkv-final-v6-2.1-3b'
args.n_layer = 32
args.n_embd = 2560
args.vocab_size = 65536
args.head_size = 64

context = "\nElon Musk's favorite"
# context = "\n北京"
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.3

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        self.w = torch.load(args.MODEL_NAME + '.pth', map_location='cuda')
        w = self.w
        w['emb.weight'] = F.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
        
        keys = list(w.keys())
        for k in keys:
            if             '.time_' in k: w[k] = w[k].squeeze()
            if k.endswith('.time_decay'): w[k] = w[k].float()
            if k.endswith('.time_faaaa'): w[k] = w[k].unsqueeze(-1).float()
        for k in keys:
            if k.endswith('maa_w'):
                w[k.replace('maa_w','maa_wkvrg')] = torch.concat([w[k],w[k.replace('maa_w','maa_k')],w[k.replace('maa_w','maa_v')],w[k.replace('maa_w','maa_r')],w[k.replace('maa_w','maa_g')]]).clone().reshape(5, -1)
                del w[k]
                del w[k.replace('maa_w','maa_k')]
                del w[k.replace('maa_w','maa_v')]
                del w[k.replace('maa_w','maa_r')]
                del w[k.replace('maa_w','maa_g')]
                
        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        assert self.head_size == args.head_size

    @MyFunction
    def channel_mixing(self, x, x_prev, time_maa_k, time_maa_r, kw, vw, rw):
        sx = x_prev - x
        k = x + sx * time_maa_k
        r = x + sx * time_maa_r

        r = torch.sigmoid(rw @ r)
        k = torch.relu(kw @ k) ** 2

        return r * (vw @ k), x

    @MyFunction
    def time_mixing(self, x, x_prev, state, maa_x, maa_wkvrg, tm_w1, tm_w2, td_w1, td_w2, time_faaaa, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        N = self.head_size

        sx = x_prev - x
        xxx = x + sx * maa_x                            # C
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)    # C @ C*5L => 5L => 5*1*L
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)         # 5*1*L @ 5*L*C => 5*1*C => 5*C
        xxx = xxx + maa_wkvrg
        xxx = xxx * sx.expand(5, -1) + x.expand(5, -1)
        w, k, v, r, g = xxx.unbind(dim=0)

        w = torch.tanh(w @ td_w1) @ td_w2
        w = w.float() + time_decay
        # assert w.dtype == torch.float
        w = torch.exp(-torch.exp(w))

        k = (kw @ k).view(H, N, 1)
        v = (vw @ v).view(H, 1, N)
        r = (rw @ r).view(H, 1, N)
        g = F.silu(gw @ g)

        kv = (k @ v).float()
        out = r @ (time_faaaa * kv + state).to(torch.bfloat16)

        state = kv + w.view(H, N, 1) * state

        out = F.group_norm(out.view(1, H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N) # same as gn(x/8, eps=1e-5)
        return ow @ (out * g), x, state

    @MyFunction
    def forward(self, token:int, state:List[torch.Tensor]):
        with torch.no_grad():
            x = self.w['emb.weight'][token]
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=self.w[bbb+'ln1.weight'], bias=self.w[bbb+'ln1.bias'])
                xx, x_out, s_out = self.time_mixing(xx, state[i*3+0], state[i*3+1],
                    self.w[att+'time_maa_x'], self.w[att+'time_maa_wkvrg'], self.w[att+'time_maa_w1'], self.w[att+'time_maa_w2'],
                    self.w[att+'time_decay_w1'], self.w[att+'time_decay_w2'], self.w[att+'time_faaaa'], self.w[att+'time_decay'],
                    self.w[att+'key.weight'], self.w[att+'value.weight'], self.w[att+'receptance.weight'], self.w[att+'gate.weight'], self.w[att+'output.weight'],
                    self.w[att+'ln_x.weight'], self.w[att+'ln_x.bias'])
                x = x + xx
                state[i*3+0] = x_out
                state[i*3+1] = s_out
                
                xx = F.layer_norm(x, (self.n_embd,), weight=self.w[bbb+'ln2.weight'], bias=self.w[bbb+'ln2.bias'])
                xx, x_out = self.channel_mixing(xx, state[i*3+2],
                    self.w[ffn+'time_maa_k'], self.w[ffn+'time_maa_r'], 
                    self.w[ffn+'key.weight'], self.w[ffn+'value.weight'], self.w[ffn+'receptance.weight'])
                x = x + xx
                state[i*3+2] = x_out
            
            x = F.layer_norm(x, (self.n_embd,), weight=self.w['ln_out.weight'], bias=self.w['ln_out.bias'])
            x = self.w['head.weight'] @ x
            return x, state

print(f'\nUsing CUDA bf16. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')

init_state = [None for _ in range(args.n_layer * 3)]
for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
    init_state[i*3+0] = torch.zeros(args.n_embd, dtype=torch.bfloat16, requires_grad=False, device="cuda")
    init_state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    init_state[i*3+2] = torch.zeros(args.n_embd, dtype=torch.bfloat16, requires_grad=False, device="cuda")

for token in tokenizer.encode(context):
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), copy.deepcopy(init_state)

    min_time = 1e10
    min_time_all = 1e10

    for i in range(LENGTH_PER_TRIAL):
        t00 = time.time()
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        t0 = time.time()

        out, state = model.forward(token, state)
        
        t1 = time.time()
        min_time = min(min_time, t1 - t0)
        min_time_all = min(min_time_all, t1 - t00)
    
    print(f'\n[{round(1/min_time_all,2)} (real) / {round(1/min_time,2)} token/s (ignore tokenizer & sampling)]', end='')

print('\n')
