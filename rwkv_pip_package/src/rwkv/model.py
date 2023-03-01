########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, gc, os, time
import torch
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

current_path = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = torch.nn.Module
    def __nop(ob):
        return ob
    MyFunction = __nop

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    wkv_cuda = load(name=f"wkv_cuda", sources=[f"{current_path}/cuda/wkv_op.cpp", f"{current_path}/cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"])

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, T, C, w, u, k, v, aa, bb, pp):
            assert 1 * C % min(C, 32) == 0
            dtype = k.dtype
            w = w.float().contiguous()
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
            y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.float)
            wkv_cuda.forward(1, T, C, w, u, k, v, y, aa, bb, pp)
            return y.to(dtype=dtype), aa, bb, pp
    def RUN_CUDA(T, C, w, u, k, v, aa, bb, pp):
        return WKV.apply(T, C, w, u, k, v, aa, bb, pp)
else:
    os.environ["RWKV_CUDA_ON"] = '0'

########################################################################################################

class RWKV(MyModule):
    def __init__(self, model, strategy):
        super().__init__()
        self.args = types.SimpleNamespace()
        args = self.args
        args.MODEL_NAME = model

        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid overflow)
        self.RESCALE_LAYER = 6 if 'fp16' in strategy else 0
        print(f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.RESCALE_LAYER}\n')

        # We will load model to CPU first
        args.MODEL_NAME = args.MODEL_NAME.strip()
        if not args.MODEL_NAME.endswith('.pth'):
            args.MODEL_NAME += '.pth'
        print(f'Loading {args.MODEL_NAME} ...')
        with torch.no_grad():
            self.w = torch.load(args.MODEL_NAME, map_location='cpu')
            w = self.w
            args.n_embd = w['emb.weight'].shape[1]
            try: # precompute embedding
                w['emb.weight'] = F.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
            except:
                w['emb.weight'] = F.layer_norm(w['emb.weight'].float(), (args.n_embd,), weight=w['blocks.0.ln0.weight'].float(), bias=w['blocks.0.ln0.bias'].float())
            del w['blocks.0.ln0.weight']
            del w['blocks.0.ln0.bias']

            keys = list(w.keys())
            args.n_layer = 0
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, layer_id+1)

            # Compute strategy
            s = [x.strip().split(' ') for x in strategy.split('->')]
            plan = [0] * len(s)
            stream_i = -1
            stream_count = 0
            to_allocate = args.n_layer + 1
            allocated = 0
            free_slots = 0
            for i in range(len(s)):
                si = s[i]
                if si[1] == 'fp32': si[1] = torch.float
                elif si[1] == 'fp16': si[1] = torch.float16
                elif si[1] == 'bf16': si[1] = torch.bfloat16
                if len(si) > 2:
                    ss = si[2]
                    assert ss.startswith('*')
                    if ss.endswith('+'):
                        plan[i] = int(ss[1:-1])
                        stream_i = i
                    else:
                        plan[i] = int(ss[1:])
                    allocated += plan[i]
                    if allocated >= to_allocate:
                        plan[i] += to_allocate - allocated
                        break
                else:
                    free_slots += 1
            if stream_i < 0:
                if free_slots > 0 and to_allocate > allocated:
                    for i in range(len(s)):
                        if plan[i] == 0:
                            plan[i] = (to_allocate - allocated) // free_slots
                            allocated += plan[i]
                            free_slots -= 1
                if to_allocate > allocated:
                    plan[len(s)-1] += to_allocate - allocated
            else:
                if to_allocate > allocated:
                    stream_count = to_allocate - allocated
                    plan[stream_i] += stream_count
            print(f'Strategy: (total {args.n_layer}+1={args.n_layer+1} layers)')
            for i in range(len(s)):
                ss = s[i]
                if i != stream_i:
                    print(f'* {ss[0]} {ss[1]}, store {plan[i]} layers')
                else:
                    print(f'* {ss[0]} {ss[1]}, store {plan[i]-stream_count} layers, stream {stream_count} layers')
                plan[i] += (0 if i == 0 else plan[i-1])
            self.strategy = [None] * (args.n_layer + 1)
            strategy = self.strategy
            for n in range(args.n_layer + 1):
                for i in range(len(s)):
                    if n < plan[i]:
                        strategy[n] = types.SimpleNamespace()
                        strategy[n].device = s[i][0]
                        strategy[n].dtype = s[i][1]
                        strategy[n].stream = False
                        if i == stream_i and n >= (plan[i] - stream_count):
                            strategy[n].stream = True
                        break
                print(f"{n}-{strategy[n].device}-{str(strategy[n].dtype).replace('torch.','')}{'-stream' if strategy[n].stream else ''}",end=' ')
            print()

            # Load weights
            print_need_newline = False
            for x in keys:
                w[x].requires_grad = False
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                if ('ln_out.' in x) or ('head.' in x):
                    layer_id = args.n_layer
                dd = strategy[layer_id]
                DEVICE = dd.device
                DTYPE = dd.dtype
                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x or 'head.weight' in x:
                    w[x] = w[x].t()
                
                if '.time_decay' in x: # need fp32 for this
                    w[x] = -torch.exp(w[x].float())
                elif '.time_first' in x: # need fp32 for this
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].to(dtype=DTYPE)

                if self.RESCALE_LAYER > 0:
                    if 'att.output.weight' in x:
                        w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                    if 'ffn.value.weight' in x:
                        w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                
                if 'emb.' in x:
                    pass
                elif (dd.stream) and (('key.weight' in x) or ('value.weight' in x) or ('receptance.weight' in x) or ('output.weight' in x)):
                    try:
                        w[x] = w[x].pin_memory() # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
                    except:
                        print('Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower.')
                elif DEVICE != 'cpu':
                    w[x] = w[x].to(device=DEVICE)

                shape = [i for i in w[x].shape if i != 1]
                if len(shape) > 1:
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f" {str(shape[0]).rjust(5)}      "
                if layer_id == 0 or layer_id >= args.n_layer-1:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    dt = str(w[x].dtype).replace('torch.', '')
                    dt = dt.replace('float32', 'fp32').replace('bfloat16', 'bf16').replace('float16', 'fp16')
                    print(x.ljust(32), dt, str(w[x].device).rjust(8), shape, ' (pinned)' if w[x].is_pinned() else '')
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)
            assert len(keys) == 4 + (4+9+5) * args.n_layer, 'Error: not a RWKV-4 model (4a and 4b models are not supported as of now)'
            gc.collect()
        
    @MyFunction
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        k = torch.square(torch.relu(kx @ kw))
        out = r * (k @ vw)
        return x + out, xx
    
    @MyFunction
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        xk = xx * k_mix + sx * (1 - k_mix)
        xr = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        out = r * (k @ vw)
        return x + out, xx[-1,:]

    @MyFunction
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        k = (kx @ kw).float()
        v = (vx @ vw).float()

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=r.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        out = (r * wkv) @ ow
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    @MyFunction
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        k = (kx @ kw).float()
        v = (vx @ vw).float()

        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=r.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        out = (r * sx) @ ow
        return x + out, xx[-1,:], aa, bb, pp
    
    @MyFunction
    def cuda_att_pre(self, x, sx, ln_w, ln_b, k_mix, v_mix, r_mix, kw, vw, rw):
        T, C = x.size()
        xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        r = torch.sigmoid(rx @ rw)
        k = kx @ kw
        v = vx @ vw
        return xx, r, k, v
    def cuda_att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        T, C = x.size()
        xx, r, k, v = self.cuda_att_pre(x, sx, ln_w, ln_b, k_mix, v_mix, r_mix, kw, vw, rw)
        y, aa, bb, pp = RUN_CUDA(T, C, t_decay, t_first, k, v, aa, bb, pp)
        out = (r * y) @ ow
        return x + out, xx[-1,:], aa, bb, pp

    def forward(self, tokens, state, full_output=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            if state == None:
                state = [None] * args.n_layer * 5
                for i in range(args.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                    dd = self.strategy[i]
                    dev = dd.device
                    dtype = dd.dtype
                    state[i*5+0] = torch.zeros(args.n_embd, dtype=dtype, requires_grad=False, device=dev)
                    state[i*5+1] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev)
                    state[i*5+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev)
                    state[i*5+3] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev) - 1e30
                    state[i*5+4] = torch.zeros(args.n_embd, dtype=dtype, requires_grad=False, device=dev)

            seq_mode = len(tokens) > 1

            x = w['emb.weight'][tokens if seq_mode else tokens[0]]

            for i in range(args.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                dd = self.strategy[i]
                dev = dd.device
                dtype = dd.dtype
                if seq_mode:
                    if 'cuda' in str(dev) and os.environ["RWKV_CUDA_ON"] == '1':
                        ATT = self.cuda_att_seq
                    else:
                        ATT = self.att_seq
                    FFN = self.ffn_seq
                else:
                    ATT = self.att_one
                    FFN = self.ffn_one

                x = x.to(dtype=dtype, device=dev)
                if dd.stream:
                    kw = w[f'{att}key.weight'].to(device=dev, non_blocking=True)
                    vw = w[f'{att}value.weight'].to(device=dev, non_blocking=True)
                    rw = w[f'{att}receptance.weight'].to(device=dev, non_blocking=True)
                    ow = w[f'{att}output.weight'].to(device=dev, non_blocking=True)
                    x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                        x, sx=state[i*5+0], aa=state[i*5+1], bb=state[i*5+2], pp=state[i*5+3],
                        ln_w=w[f'{bbb}ln1.weight'], ln_b=w[f'{bbb}ln1.bias'],
                        k_mix=w[f'{att}time_mix_k'], v_mix=w[f'{att}time_mix_v'], r_mix=w[f'{att}time_mix_r'],
                        t_decay = w[f'{att}time_decay'], t_first = w[f'{att}time_first'],
                        kw=kw, vw=vw, rw=rw, ow=ow)
                    del kw
                    del vw
                    del rw
                    del ow
                else:
                    x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                        x, sx=state[i*5+0], aa=state[i*5+1], bb=state[i*5+2], pp=state[i*5+3],
                        ln_w=w[f'{bbb}ln1.weight'], ln_b=w[f'{bbb}ln1.bias'],
                        k_mix=w[f'{att}time_mix_k'], v_mix=w[f'{att}time_mix_v'], r_mix=w[f'{att}time_mix_r'],
                        t_decay = w[f'{att}time_decay'], t_first = w[f'{att}time_first'],
                        kw=w[f'{att}key.weight'],
                        vw=w[f'{att}value.weight'],
                        rw=w[f'{att}receptance.weight'],
                        ow=w[f'{att}output.weight'])
                if dd.stream:
                    kw = w[f'{ffn}key.weight'].to(device=dev, non_blocking=True)
                    vw = w[f'{ffn}value.weight'].to(device=dev, non_blocking=True)
                    rw = w[f'{ffn}receptance.weight'].to(device=dev, non_blocking=True)
                    x, state[i*5+4] = FFN(
                        x, sx=state[i*5+4],
                        ln_w=w[f'{bbb}ln2.weight'], ln_b=w[f'{bbb}ln2.bias'],
                        k_mix=w[f'{ffn}time_mix_k'], r_mix=w[f'{ffn}time_mix_r'],
                        kw=kw, vw=vw, rw=rw)
                    del kw
                    del vw
                    del rw
                else:
                    x, state[i*5+4] = FFN(
                        x, sx=state[i*5+4],
                        ln_w=w[f'{bbb}ln2.weight'], ln_b=w[f'{bbb}ln2.bias'],
                        k_mix=w[f'{ffn}time_mix_k'], r_mix=w[f'{ffn}time_mix_r'],
                        kw=w[f'{ffn}key.weight'],
                        vw=w[f'{ffn}value.weight'],
                        rw=w[f'{ffn}receptance.weight'])

                if self.RESCALE_LAYER > 0:
                    if (i+1) % self.RESCALE_LAYER == 0:
                        x = x / 2
            
            x = x[-1,:] if (seq_mode and (not full_output)) else x
            x = x.to(dtype=self.strategy[args.n_layer].dtype, device=self.strategy[args.n_layer].device)
            x = F.layer_norm(x, (args.n_embd,), weight=w['ln_out.weight'], bias=w['ln_out.bias'])
            x = x @ w['head.weight']

            return x.float(), state
