
from rwkv.model import RWKV
from gptq.datautils import *
from gptq.quant import Quantizer, quantize

import os
import torch.nn.functional as F
import torch.nn as nn
import time
import gc
import math
import re

class GPTQ_RWKV(RWKV):

    ### begin GPTQ
    class GPTQ:
        def __init__(self, weight, name):
            #TODO: Remove name, only used for debugging
            self.name = name
            self.weight = weight.clone()
            self.dev = weight.device
            # In GPTQ, they use nn.Linear(x) which performs x @ w.T but in RWKV, we perform x @ w instead
            # Problem is self.H is a square matrix which depends on self.columns = W.shape[1] in the original code
            # But if we keep it that way, this will break self.H += inp.matmul(inp.t()) because inp.shape[1] != W.shape[1]
            # Thus, we have to use self.W.shape[0] instead
            self.columns = self.weight.shape[0]
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
            self.nsamples = 0

        def add_batch(self, inp):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            
            #TODO: is the case with len = 1 still necessary ?
            tmp = 1 if len(inp.shape) == 1 else inp.shape[0]

            # Assume weight come from nn.Linear
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

        def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
            W = self.weight.data.clone()
            # Need to transpose here, same reason as in __init__ with self.columns
            W = W.t()
            W = W.float()

            tick = time.time()

            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

            H = self.H
            del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()
            print('time %.2f' % (time.time() - tick))
            print('error', torch.sum(Losses).item())

            if actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]

            self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)

    ### end GPTQ

    ### begin GPTQ_RWKV
    def __init__(self, model, strategy):
        super().__init__(model, strategy)
        #TODO: add assert to only quantize in CPU FP32 mode

    def _fill_subset(self, layer_id):
        # Keep only layer within block layer_id
        dd = self.strategy[layer_id]
        dev = dd.device

        for name in self.w.keys():
            if re.match(f'^blocks\.{layer_id}\..*\.weight$', name):
                tensor = self.w[name]

                #TODO: Skip 1D tensors for now
                if len(tensor.shape) == 1:
                    continue
                
                print(f"{name} = {self.w[name].shape}")
                    
                if re.match(f'^blocks\.{layer_id}\.(?:att|ffn)\.(?:key|value|output|receptance)\.weight$', name):
                    tensor = tensor.to(device=dev, non_blocking=True)

                self.subset[name] = tensor
    
    def alloc_gptq(self, layer_id):
        self.subset = {}
        self.gptq = {}

        self._fill_subset(layer_id)
        
        for name in self.subset:
            self.gptq[name] = self.GPTQ(self.subset[name], name)
            self.gptq[name].quantizer = Quantizer()
            #TODO: add argparse to configure
            self.gptq[name].quantizer.configure(bits=4, perchannel=True, sym=False, mse=False, trits=False)

    def free_gptq(self):
        self.subset = {}
        self.gptq = {}

    def fasterquant(self, layer_id, quantizers):

        for name in self.subset:
            print(f"Quantizing {name} of layer {layer_id}")
            #TODO: add argparse to fastquant
            self.gptq[name].fasterquant(percdamp=0.01, groupsize=-1, actorder=False)
            # self.gptq[name].fastquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
            quantizers[name] = self.gptq[name].quantizer
            # TODO: may be free gptq here to save memory

    ### end GPTQ_RWKV

    ### begin RWKV
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw.weight)
        rw.add_batch(rx)
        k = (kx @ kw.weight).float()
        kw.add_batch(kx)
        v = (vx @ vw.weight).float()
        vw.add_batch(vx)

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        out = (r * wkv) @ ow.weight
        ow.add_batch((r * wkv))
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p
    
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw.weight)
        rw.add_batch(rx)
        k = (kx @ kw.weight).float()
        kw.add_batch(kx)
        v = (vx @ vw.weight).float()
        vw.add_batch(vx)

        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        out = (r * sx) @ ow.weight
        ow.add_batch((r * sx))
        return x + out, xx[-1,:], aa, bb, pp

    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw.weight)
        rw.add_batch(rx)
        vx = torch.square(torch.relu(kx @ kw.weight))
        kw.add_batch(kx)
        out = r * (vx @ vw.weight)
        vw.add_batch(vx)
        return x + out, xx

    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # x = (2048, 768)
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # xx = (2048, 768)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # sx = (2048, 768)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        # kx = (2048, 768)
        # rx = (2048, 768)

        r = torch.sigmoid(rx @ rw.weight)
        # r = (2048, 768)
        rw.add_batch(rx)
        vx = torch.square(torch.relu(kx @ kw.weight))
        # vx = (2048, 3072)
        # kx: (2048, 768)
        # kw.weight: (768, 3072)
        # vx: (2048, 3072)
        kw.add_batch(kx)
        out = r * (vx @ vw.weight)
        vw.add_batch(vx)
        return x + out, xx[-1,:]

    def forward_block(self, x, state, i, seq_mode, full_output=False):
        with torch.no_grad():
            args = self.args

            if state == None:
                state = [None] * args.n_layer * 5
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                state[i*5+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                state[i*5+1] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i*5+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i*5+3] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
                state[i*5+4] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'
            dd = self.strategy[i]
            dev = dd.device
            atype = dd.atype
            wtype = dd.wtype

            if seq_mode:
                if 'cuda' in str(dev) and os.environ["RWKV_CUDA_ON"] == '1':
                    ATT = self.cuda_att_seq if wtype != torch.uint8 else self.cuda_att_seq_i8
                else:
                    ATT = self.att_seq if wtype != torch.uint8 else self.att_seq_i8
                FFN = self.ffn_seq if wtype != torch.uint8 else self.ffn_seq_i8
            else:
                ATT = self.att_one if wtype != torch.uint8 else self.att_one_i8
                FFN = self.ffn_one if wtype != torch.uint8 else self.ffn_one_i8

            x = x.to(dtype=atype, device=dev)

            kw = self.gptq[f'{att}key.weight']
            vw = self.gptq[f'{att}value.weight']
            rw = self.gptq[f'{att}receptance.weight']
            ow = self.gptq[f'{att}output.weight']

            kmx = self.w[f'{att}key.weight_mx'] if wtype == torch.uint8 else x
            krx = self.w[f'{att}key.weight_rx'] if wtype == torch.uint8 else x
            kmy = self.w[f'{att}key.weight_my'] if wtype == torch.uint8 else x
            kry = self.w[f'{att}key.weight_ry'] if wtype == torch.uint8 else x
            vmx = self.w[f'{att}value.weight_mx'] if wtype == torch.uint8 else x
            vrx = self.w[f'{att}value.weight_rx'] if wtype == torch.uint8 else x
            vmy = self.w[f'{att}value.weight_my'] if wtype == torch.uint8 else x
            vry = self.w[f'{att}value.weight_ry'] if wtype == torch.uint8 else x
            rmx = self.w[f'{att}receptance.weight_mx'] if wtype == torch.uint8 else x
            rrx = self.w[f'{att}receptance.weight_rx'] if wtype == torch.uint8 else x
            rmy = self.w[f'{att}receptance.weight_my'] if wtype == torch.uint8 else x
            rry = self.w[f'{att}receptance.weight_ry'] if wtype == torch.uint8 else x
            omx = self.w[f'{att}output.weight_mx'] if wtype == torch.uint8 else x
            orx = self.w[f'{att}output.weight_rx'] if wtype == torch.uint8 else x
            omy = self.w[f'{att}output.weight_my'] if wtype == torch.uint8 else x
            ory = self.w[f'{att}output.weight_ry'] if wtype == torch.uint8 else x
        
            x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                x=x, sx=state[i*5+0], aa=state[i*5+1], bb=state[i*5+2], pp=state[i*5+3],
                ln_w=self.w[f'{bbb}ln1.weight'], ln_b=self.w[f'{bbb}ln1.bias'],
                k_mix=self.w[f'{att}time_mix_k'], v_mix=self.w[f'{att}time_mix_v'], r_mix=self.w[f'{att}time_mix_r'],
                t_decay=self.w[f'{att}time_decay'], t_first=self.w[f'{att}time_first'],
                kw=kw, vw=vw, rw=rw, ow=ow,
                kmx=kmx, krx=krx, kmy=kmy, kry=kry,
                vmx=vmx, vrx=vrx, vmy=vmy, vry=vry,
                rmx=rmx, rrx=rrx, rmy=rmy, rry=rry,
                omx=omx, orx=orx, omy=omy, ory=ory,
                )

            if dd.stream:
                del kw, vw, rw, ow

            kw = self.gptq[f'{ffn}key.weight']
            vw = self.gptq[f'{ffn}value.weight']
            rw = self.gptq[f'{ffn}receptance.weight']
            if dd.stream:
                kw = kw.to(device=dev, non_blocking=True)
                vw = vw.to(device=dev, non_blocking=True)
                rw = rw.to(device=dev, non_blocking=True)

            kmx = self.w[f'{ffn}key.weight_mx'] if wtype == torch.uint8 else x
            krx = self.w[f'{ffn}key.weight_rx'] if wtype == torch.uint8 else x
            kmy = self.w[f'{ffn}key.weight_my'] if wtype == torch.uint8 else x
            kry = self.w[f'{ffn}key.weight_ry'] if wtype == torch.uint8 else x
            vmx = self.w[f'{ffn}value.weight_mx'] if wtype == torch.uint8 else x
            vrx = self.w[f'{ffn}value.weight_rx'] if wtype == torch.uint8 else x
            vmy = self.w[f'{ffn}value.weight_my'] if wtype == torch.uint8 else x 
            vry = self.w[f'{ffn}value.weight_ry'] if wtype == torch.uint8 else x 
            rmx = self.w[f'{ffn}receptance.weight_mx'] if wtype == torch.uint8 else x
            rrx = self.w[f'{ffn}receptance.weight_rx'] if wtype == torch.uint8 else x
            rmy = self.w[f'{ffn}receptance.weight_my'] if wtype == torch.uint8 else x
            rry = self.w[f'{ffn}receptance.weight_ry'] if wtype == torch.uint8 else x
            x, state[i*5+4] = FFN(
                x=x, sx=state[i*5+4],
                ln_w=self.w[f'{bbb}ln2.weight'], ln_b=self.w[f'{bbb}ln2.bias'],
                k_mix=self.w[f'{ffn}time_mix_k'], r_mix=self.w[f'{ffn}time_mix_r'],
                kw=kw, vw=vw, rw=rw,
                kmx=kmx, krx=krx, kmy=kmy, kry=kry,
                vmx=vmx, vrx=vrx, vmy=vmy, vry=vry,
                rmx=rmx, rrx=rrx, rmy=rmy, rry=rry,                    
                )
            
            if dd.stream:                
                del kw, vw, rw
            
            if self.RESCALE_LAYER > 0:
                if (i+1) % self.RESCALE_LAYER == 0:
                    x = x / 2
        
        dd = self.strategy[args.n_layer]
        x = x[-1,:] if (seq_mode and (not full_output)) else x
        x = x.to(dtype=dd.atype, device=dd.device)
        
        #TODO: Add GPTQ support for head & ln_out
        x = F.layer_norm(x, (args.n_embd,), weight=self.w['ln_out.weight'], bias=self.w['ln_out.bias'])
        if self.w['head.weight'].dtype != torch.uint8:
            x = x @ self.w['head.weight']
        else:
            if seq_mode and full_output:
                x = self.mm8_seq(x, self.w['head.weight'], self.w['head.weight_mx'], self.w['head.weight_rx'], self.w['head.weight_my'], self.w['head.weight_ry'])
            else:
                x = self.mm8_one(x, self.w['head.weight'], self.w['head.weight_mx'], self.w['head.weight_rx'], self.w['head.weight_my'], self.w['head.weight_ry'])

        return x.float(), state
    
    ### end RWKV

NSAMPLES=2
HIDDEN_SIZE=768
SEQLEN=2048 # TODO: this is chosen by the model

# train_tokens, test_tokens = get_loaders(
#     dataset_name="wikitext2",
#     nsamples=NSAMPLES,
#     seed=42,
#     seqlen=SEQLEN,
#     model=None
# )

# tokens = torch.cat([inp for inp, _ in train_tokens], dim=0)
tokens = torch.zeros((NSAMPLES, SEQLEN), dtype=torch.int64)
print("tokens.shape", tokens.shape)

model = GPTQ_RWKV("./RWKV-4-Pile-169M-20220807-8023.pth", strategy='cpu fp32')

#TODO: Do the same in GPU side
with torch.no_grad():
    seq_mode = len(tokens) > 1
    x = model.w['emb.weight'][tokens if seq_mode else tokens[0]]

    quantizers = {}

    for layer_id in range(model.args.n_layer):

        model.alloc_gptq(layer_id)

        for j in range(NSAMPLES):
            _ = model.forward_block(x[j], state=None, i=layer_id, seq_mode=seq_mode)
            
        model.fasterquant(layer_id, quantizers)

        model.free_gptq()

        #TODO: Since we quantize per block, we should pass the outputs of block 0 to input of block 1 ?
        # inps, outs = outs, inps

# TODO: create a function that check if all weights were properly quantized
print("Done")