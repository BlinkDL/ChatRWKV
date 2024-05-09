########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time
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

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

########################################################################################################

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).cpu().numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

########################################################################################################

# cuda bf16 inference

tokenizer = RWKV_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

args = types.SimpleNamespace()
args.MODEL_NAME = '/home/rwkv/rwkv-final-v6-2.1-3b'
args.n_layer = 32
args.n_embd = 2560
args.vocab_size = 65536

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
        self.eval() # set torch to inference mode
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cuda')
        w['emb.weight'] = F.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
        
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze()
            if k.endswith('.time_decay'): w[k] = w[k].float()
            if k.endswith('.time_faaaa'): w[k] = w[k].unsqueeze(-1).float()

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head

        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_decay" => self.w.blocks[0].att.time_decay
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, x_prev, time_maa_k, time_maa_r, kw, vw, rw):
        sx = x_prev - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r
        r = torch.sigmoid(rw @ xr)
        k = torch.relu(kw @ xk) ** 2
        return r * (vw @ k), x

    @MyFunction
    def time_mixing(self, x, x_prev, s, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_faaaa, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        N = self.head_size

        sx = x_prev - x
        xxx = x + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)

        r = (rw @ xr).view(H, 1, N)
        k = (kw @ xk).view(H, N, 1)
        v = (vw @ xv).view(H, 1, N)
        g = F.silu(gw @ xg)

        a = (k @ v).float()
        out = r @ (time_faaaa * a + s).to(torch.bfloat16)

        w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, N, 1)
        assert w.dtype == torch.float
        w = torch.exp(-torch.exp(w))
        s = a + w * s

        out = F.group_norm(out.view(1, H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N) * g # same as gn(x/8, eps=1e-5)
        return ow @ out, x, s

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = [None for _ in range(args.n_layer * 3)]
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(self.args.n_embd, dtype=torch.bfloat16, requires_grad=False, device="cuda")
                    state[i*3+1] = torch.zeros((self.n_head, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                    state[i*3+2] = torch.zeros(self.args.n_embd, dtype=torch.bfloat16, requires_grad=False, device="cuda")
            
            x = self.w.emb.weight[token]
            for i in range(self.args.n_layer):
                
                att = self.w.blocks[i].att
                xx, saa, ss = self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state[i*3+0], state[i*3+1],
                    att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                    att.time_decay_w1, att.time_decay_w2, att.time_faaaa, att.time_decay,
                    att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias)
                x = x + xx
                state[i*3+0] = saa
                state[i*3+1] = ss
                
                ffn = self.w.blocks[i].ffn
                xx, ss = self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state[i*3+2],
                    ffn.time_maa_k, ffn.time_maa_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                x = x + xx
                state[i*3+2] = ss
            
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

print(f'\nUsing CUDA bf16. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = None
for token in tokenizer.encode(context):
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), copy.deepcopy(init_state)

    min_time = 1e10

    for i in range(LENGTH_PER_TRIAL):
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
    
    print(f'\n[{round(1/min_time,2)} token/s (ignore tokenizer & sampling)]', end='')

print('\n')
