
from rwkv.model import RWKV
from gptq.gptq import *
from gptq.datautils import *
import os
import torch.nn.functional as F
import gc
import re

if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    def __nop(ob):
        return ob
    MyFunction = __nop
    MyStatic = __nop

class GPTQ_RWKV(RWKV):

    ### begin GPTQ
    class GPTQ:
        def __init__():
            pass

        def add_batch(self):
            pass

        def fasterquant(self):
            pass

    def __init__(self, model, strategy):
        super().__init__(model, strategy)

        self.subset = {}
        self.gptq = {}
    ### end GPTQ

    def _filter_layer_within_block(self, layer_id, model):

        def _create_layer(model, name):
            if len(model.w[name].shape) == 1:
                #TODO: maybe reshape (-1, 1) ?
                w = model.w[name].reshape(1, -1)
                layer = nn.Linear(*w.shape, bias=False)
                layer.weight = nn.Parameter(w)
            else:
                layer = nn.Linear(*model.w[name].shape, bias=False)
                layer.weight = nn.Parameter(model.w[name])
            return layer

        res = {}
        dd = model.strategy[layer_id]
        dev = dd.device

        for name in model.w.keys():
            if re.match(f'^blocks\.{layer_id}\..*\.weight$', name):
                layer = _create_layer(model, name)
                print(f"{name} = {model.w[name].shape}")
                
                if re.match(f'^blocks\.{layer_id}\.(?:att|ffn)\.(?:key|value|output|receptance)\.weight$', name):
                    layer = layer.to(device=dev, non_blocking=True)

                res[name] = layer

        return res
    
    def alloc_gptq(self, layer_id, subset):
        
        self.subset = self.__filter_layer_within_block(layer_id, model)
        
        for name in subset:
            self.gptq[name] = GPTQ(subset[name])
            self.gptq[name].quantizer = Quantizer()
            self.gptq[name].quantizer.configure(bits=4, perchannel=True, sym=False, mse=False, trits=False)

    def free_gptq(self):
        del self.subset
        del self.gptq
        gc.collect()

    def fasterquant(self, layer_id, quantizers):
        for name in self.subset:
            print(f"Quantizing {name} of layer {layer_id}")
            #TODO: add argparse to fasterquand
            self.gptq[name].fastquant(percdamp=0.01, groupsize=-1, actorder=False)
            # self.gptq[name].fastquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
            quantizers[name] = self.gptq[name].quantizer

    @MyFunction
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        k = (kx @ kw).float()
        # k = (kx @ kw.weight).float()
        # kw.add_batch(kx)
        v = (vx @ vw).float()

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        out = (r * wkv) @ ow
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p
    
    @MyFunction
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
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
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
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
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        vx = torch.square(torch.relu(kx @ kw))
        out = r * (vx @ vw)
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
                ln_w=self.gptq[f'{bbb}ln1.weight'], ln_b=self.w[f'{bbb}ln1.bias'],
                k_mix=self.w[f'{att}time_mix_k'], v_mix=self.w[f'{att}time_mix_v'], r_mix=self.w[f'{att}time_mix_r'],
                t_decay=self.w[f'{att}time_decay'], t_first=self.w[f'{att}time_first'],
                kw=kw, vw=vw, rw=rw, pw=ow,
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
                ln_w=self.gptq[f'{bbb}ln2.weight'], ln_b=self.w[f'{bbb}ln2.bias'],
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


NSAMPLES=1
HIDDEN_SIZE=768
SEQLEN=HIDDEN_SIZE # TODO: this is chosen by the model

train_tokens, test_tokens = get_loaders(
    dataset_name="wikitext2",
    nsamples=NSAMPLES,
    seed=42,
    seqlen=SEQLEN,
    model=None
)

tokens = [inp.squeeze() for inp, _ in train_tokens]

model = GPTQ_RWKV("./RWKV-4-Pile-169M-20220807-8023.pth", strategy='cpu fp32')

with torch.no_grad():
    seq_mode = len(tokens) > 1
    x = model.w['emb.weight'][tokens if seq_mode else tokens[0]]
    
    quantizers = {}

    for layer_id in range(model.args.n_layer):

        model.alloc_gptq(layer_id, model)

        for j in range(NSAMPLES):
            _ = model.forward_block(x[j].unsqueeze(0), state=None, i=layer_id, seq_mode=seq_mode, full_output=full_output)

        model.fasterquant(layer_id, quantizers)

        model.free_gptq()

# TODO: create a function that check if all weights were properly quantized
