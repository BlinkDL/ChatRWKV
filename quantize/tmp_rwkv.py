from myRWKV import RWKV
from gptq.datautils import *
from gptq.quant import Quantizer, quantize

import os
import torch.nn.functional as F
from collections import OrderedDict
import time
import math
import re
from gptq.gptq import QuantLinear_custom 

WBITS = 8
GROUPSIZE = -1

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
            self.deactivate_add_batch_call = False

        def add_batch(self, inp):
            # After calling fasterquant, we don't want to call add_batch anymore
            if self.deactivate_add_batch_call:
                return

            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            
            tmp = inp.shape[0]
            
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
            # OLD: Need to transpose here, same reason as in __init__ with self.columns
            # UPDATE: no need to tranpose as we already transpose in my_linear()
            # UPDATE2: for rwkv, this is necessary
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
            
            g_idx = []
            scale = []
            zero = []
            now_idx = 1

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

                        if ((i1 + i) // groupsize) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1

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
            
            groupsize = groupsize if groupsize != -1 else self.columns
            g_idx = [i // groupsize  for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
            if actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
                g_idx = g_idx[invperm]

            self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
           
            if scale == []:
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)
            scale = torch.cat(scale,dim=1)
            zero = torch.cat(zero,dim=1)
            return scale,zero,g_idx

    ### end GPTQ

    ### begin GPTQ_RWKV
    def __init__(self, checkpoint_path, strategy):
        super().__init__(checkpoint_path, strategy)
        for i in range(self.args.n_layer):
            assert self.strategy[i].device == "cpu"

    def _fill_subset(self, layer_id):
        # Keep only layer within block layer_id

        #TODO: Uncomment me when quantizing 1 layer works
        # is_weight = re.compile(f'^blocks\.{layer_id}\..*\.weight$')
        is_weight = re.compile("blocks.0.att.key.weight")
        for name in self.w.keys():                
            if is_weight.match(name):
                if len(self.w[name].shape) == 1: continue #TODO: Skip 1D tensors for now
                self.subset[name] = self.w[name]

        # TODO: Uncomment me when quantizing 1 layer works
        # is_last_layer = (layer_id == self.args.n_layer - 1)
        # if is_last_layer:
        #     self.subset["head.weight"] = self.w["head.weight"]
        
        return self.subset

    def alloc_gptq(self, layer_id):
        self.subset = {}
        self.gptq = {}

        self.subset = self._fill_subset(layer_id)

        for name in self.subset:
            self.gptq[name] = self.GPTQ(self.subset[name], name)
            self.gptq[name].quantizer = Quantizer()
            self.gptq[name].quantizer.configure(bits=WBITS, perchannel=True, sym=False, mse=False, trits=False)

    def free_gptq(self):
        self.subset = {}
        self.gptq = {}

    def fasterquant(self, layer_id, quantizers):

        for name in self.subset:
            print(layer_id, name)
            print('Quantizing ...')
            scale,zero,g_idx = self.gptq[name].fasterquant(percdamp=0.01, groupsize=GROUPSIZE, actorder=False)
            quantizers[name] = (self.gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu())

    ### end GPTQ_RWKV

    ### begin RWKV
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(rx @ rw.weight)
        r = torch.sigmoid(rx @ rw)
        # rw.add_batch(rx)

        k = (kx @ kw.weight).float()
        kw.add_batch(kx)

        # v = (vx @ vw.weight).float()
        v = (vx @ vw).float()
        # vw.add_batch(vx)

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        # out = (r * wkv) @ ow.weight
        out = (r * wkv) @ ow
        # ow.add_batch(r * wkv)
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p
    
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(rx @ rw.weight)
        r = torch.sigmoid(rx @ rw)
        # rw.add_batch(rx)
        k = (kx @ kw.weight).float()
        kw.add_batch(kx)
        # v = (vx @ vw.weight).float()
        v = (vx @ vw).float()
        # vw.add_batch(vx)

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
        # out = (r * sx) @ ow.weight
        out = (r * sx) @ ow
        # ow.add_batch(r * sx)
        return x + out, xx[-1,:], aa, bb, pp

    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(rx @ rw.weight)
        r = torch.sigmoid(rx @ rw)
        # rw.add_batch(rx)
        # vx = torch.square(torch.relu(kx @ kw.weight))
        vx = torch.square(torch.relu(kx @ kw))
        # kw.add_batch(kx)
        # out = r * (vx @ vw.weight)
        out = r * (vx @ vw)
        # vw.add_batch(vx)
        return x + out, xx

    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(rx @ rw.weight)
        r = torch.sigmoid(rx @ rw)
        # rw.add_batch(rx)
        # vx = torch.square(torch.relu(kx @ kw.weight))
        vx = torch.square(torch.relu(kx @ kw))
        # kw.add_batch(kx)
        # out = r * (vx @ vw.weight)
        out = r * (vx @ vw)
        # vw.add_batch(vx)
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
            # vw = self.gptq[f'{att}value.weight']
            vw = self.w[f'{att}value.weight']
            # rw = self.gptq[f'{att}receptance.weight']
            rw = self.w[f'{att}receptance.weight']
            # ow = self.gptq[f'{att}output.weight']
            ow = self.w[f'{att}output.weight']

            if dd.stream:
                kw = kw.to(device=dev, non_blocking=True)
                vw = vw.to(device=dev, non_blocking=True)
                rw = rw.to(device=dev, non_blocking=True)
                ow = ow.to(device=dev, non_blocking=True)
               
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

            # kw = self.gptq[f'{ffn}key.weight']
            kw = self.w[f'{ffn}key.weight']
            # vw = self.gptq[f'{ffn}value.weight']
            vw = self.w[f'{ffn}value.weight']
            # rw = self.gptq[f'{ffn}receptance.weight']
            rw = self.w[f'{ffn}receptance.weight']

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
        
        is_last_layer = i == (args.n_layer - 1)
        if is_last_layer:
            dd = self.strategy[args.n_layer]
            x = x[-1,:] if (seq_mode and (not full_output)) else x
            x = x.to(dtype=dd.atype, device=dd.device)
            
            #TODO: ln_out.weight is 1D tensor
            x = F.layer_norm(x, (args.n_embd,), weight=self.w['ln_out.weight'], bias=self.w['ln_out.bias'])

            if self.w['head.weight'].dtype != torch.uint8:
                x = x @ self.w['head.weight']
                #TODO: uncommenbt me when quantizing 1 layer work
                # x = x @ self.gptq['head.weight'].weight
                # self.gptq['head.weight'].add_batch(x)

        return x.float()
    
    ### end RWKV

@torch.no_grad()
def quantize_gptq_custom(model, tokens):
    nsamples = tokens.shape[0]
    seq_mode = len(tokens) > 1
    is_last_layer = lambda x: x == (model.args.n_layer - 1)

    inps = model.w['emb.weight'][tokens if seq_mode else tokens[0]]
    outs = torch.zeros_like(inps)
    quantizers = {}
    
    # for layer_id in range(model.args.n_layer):
    for layer_id in range(1):
        
        print(f"Quantizing layer {layer_id} ...")

        model.alloc_gptq(layer_id)

        for i in range(nsamples):
            #TODO: Are outs value normal ? (they look almost all the same)
            if not is_last_layer(layer_id):
                outs[i] = model.forward_block(inps[i], state=None, i=layer_id, seq_mode=seq_mode)
            else:
                _ = model.forward_block(inps[i], state=None, i=layer_id, seq_mode=seq_mode)

        for gptq_layer in model.gptq.values():
            gptq_layer.deactivate_add_batch_call = True

        model.fasterquant(layer_id, quantizers)

        for i in range(nsamples):
            if not is_last_layer(layer_id):
                outs[i] = model.forward_block(inps[i], state=None, i=layer_id, seq_mode=seq_mode)
            else:
                _ = model.forward_block(inps[i], state=None, i=layer_id, seq_mode=seq_mode)

        # Assign the quantized weights to the model
        for key in model.gptq.keys():
            model.w[key].copy_(model.gptq[key].weight)

        model.free_gptq()

        # We need to pass the outputs of block i as input of block i+1 (except for last block)
        if not is_last_layer(layer_id):
            inps, outs = outs, inps

    return quantizers

def model_pack_custom(model, quantizers, wbits, groupsize):

    weights = OrderedDict()

    # is_weight = re.compile('^blocks\.\d+(\.[a-z]+[0-9]?)*\.weight$')
    # for name in model.w.keys():                
    #     if is_weight.match(name):
    #         if len(model.w[name].shape) == 1: continue #TODO: Skip 1D tensors for now
    #         weights[name] = model.w[name]
    
    for name in quantizers.keys():
        if len(model.w[name].shape) == 1: continue
        weights[name] = model.w[name]

    #TODO: uncommenbt me when done
    # weights["head.weight"] = model.w["head.weight"]

    assert set(quantizers) - set(weights) == set(), "Quantizers and weights don't match"
    assert set(weights) - set(quantizers) == set(), "Quantizers and weights don't match"

    # Replace layer by QuantLinear
    model.w_quant = {}
    for key, value in model.w.items():
        if key in quantizers.keys():
            #FIXME: So far, we don't quantize ln0 et ln1 (which have bias) because 1d tensors
            bias = None
            model.w_quant[key] = QuantLinear_custom(wbits, groupsize, value.shape[0], value.shape[1], bias)

    # Fill QuantLinear
    print('Packing ...')
    for key in model.w_quant.keys():
        _, scale,zero,g_idx = quantizers[key]
        bias = None
        model.w_quant[key].pack(weights[key], bias, scale, zero, g_idx)
    print('Done.')
    return model


if __name__ == "__main__":

    model_ref = GPTQ_RWKV("./RWKV-4-Pile-169M-20220807-8023.pth", strategy='cpu fp32')
    model = GPTQ_RWKV("./RWKV-4-Pile-169M-20220807-8023.pth", strategy='cpu fp32')

    NSAMPLES=1
    HIDDEN_SIZE=model.args.n_embd
    SEQLEN=1024 # cf https://huggingface.co/BlinkDL/rwkv-4-pile-169m

    train_tokens, test_tokens = get_loaders(
        dataset_name="wikitext2",
        nsamples=NSAMPLES,
        seed=42,
        seqlen=SEQLEN,
        model=model
    )

    tokens = torch.cat([inp for inp, _ in train_tokens], dim=0)
    tokens = torch.zeros((NSAMPLES, SEQLEN), dtype=torch.int64)
    print("tokens.shape", tokens.shape)
    
    quantizers = quantize_gptq_custom(model, tokens)
    model = model_pack_custom(model, quantizers, WBITS, GROUPSIZE)
    torch.save([model.w_quant, model.w], "1sample_quantized.pth")
    
    # Make sure only 1 layer was quantized
    assert len(model.w_quant.keys()) == 1 and "blocks.0.att.key.weight" in model.w_quant.keys()

    for (ref_key, ref_value), (key, value) in zip(model_ref.w.items(), model.w.items()):
        if key != "blocks.0.att.key.weight":
            assert torch.allclose(ref_value, value, atol=1e-5)
        else:
            assert not torch.allclose(ref_value, value, atol=1e-5)

    print("Done Custom GPTQ")

    # I have noticed QuantLinear.forward() can be divded in 2 parts:
    # 1. Quantize the weights (using info from model.w_quant thanks to QuantLinear.pack())
    # 2. Perform x @ weights
    # We can load checkpoint  RWKV of base class with model_w (which are quantized but doesnt have the scale, zero info)
    # Then, if isinstancce(model, w_quant) exist, we load this dict as well
    # Each time the weights are called, we do a trigger() by checking if isinstancce(moded.w_quant is QuantLinear) 
    # This way, we can reuse RWKV base class with minimal change 