import argparse
import time
import re
import torch
import torch.nn as nn
import torch.optim as optim

from sanity_check_utils import seed_everything, MNISTloader, SimpleNet, train, evaluate, SimpleNet_V2
from gptq import *
from modelutils import *
from quant import *

WBITS = 8
GROUPSIZE = -1

## =============== REFERENCE ===============
def model_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name],scale,zero,g_idx = quantizers[name] 
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model

def load_quant(model, checkpoint, wbits, groupsize):
    print('Loading model ...')
    model = model.eval()
    layers = find_layers(model)

    # Don't quantize the last layer because qzeros is empty (I don't know why they create qzeros that way)
    # (gptq.py:L235, second dimension of qzeros is 0 because last layer is 10 for classification)
    for name in ["linear4"]:
        if name in layers:
            del layers[name]

    make_quant(model, layers, wbits, groupsize)    
    model.load_state_dict(torch.load(checkpoint))
    print('Done.')
    return model

@torch.no_grad()
def quantize_gptq(model, train_loader):
    quantizers = {}
    layers = list(model.modules())[1:]
    layers = [l for l in layers if isinstance(l, nn.Linear)]
    layers = layers[:-1]
    is_last_layer = lambda x: x == (len(layers) - 1)

    nsamples = len(train_loader.dataset)
    batch_size = train_loader.batch_size

    inps = torch.zeros((nsamples, model.N), dtype=torch.float)
    for i, (inp, _) in enumerate(train_loader):
        inps[i*batch_size:(i+1)*batch_size] = inp.view(-1, 32*32)
    outs = torch.zeros_like(inps)
    

    for layer_id in range(len(layers)):
        layer = layers[layer_id]

        subset = find_layers(layer)
        gptq = {}

        for name in subset:
            gptq[name] = GPTQ(subset[name], name)
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(bits=WBITS, perchannel=True, sym=True, mse=False, trits=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for i in range(nsamples):
            if not is_last_layer(layer_id):
                outs[i] = layer(inps[i])
            else:
                _ = layer(inps[i])

        for h in handles: h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            scale,zero,g_idx = gptq[name].fasterquant(percdamp=0.01, groupsize=GROUPSIZE, actorder=False)
            quantizers[f"linear{layer_id + 1}"] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu())
            gptq[name].free()

        for i in range(nsamples):
            if not is_last_layer(layer_id):
                outs[i] = layer(inps[i])
            else:
                _ = layer(inps[i])
                
        del layer
        del gptq 
        torch.cuda.empty_cache()

        if not is_last_layer(layer_id):
            inps, outs = outs, inps
    
    return quantizers

## =============== OUR IMPLEMENTATION ===============
class GPTQ_CUSTOM(SimpleNet_V2):

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

            #TODO: Do we have to uncomment it ?
            # if isinstance(self.layer, transformers.Conv1D):
            #     Q = Q.t()
            self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
           
            if scale == []:
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)
            scale = torch.cat(scale,dim=1)
            zero = torch.cat(zero,dim=1)
            return scale,zero,g_idx

    ### end GPTQ

    ### begin GPTQ_CUSTOM
    def __init__(self, checkpoint_path):
        super().__init__()
        self.load_state_dict(torch.load(checkpoint_path,  map_location="cpu"))        
    
    def _fill_subset(self, layer_id):
        is_last_layer = (layer_id == self.nb_layers - 1)
        if is_last_layer:
            return {}
        # Keep only layer within block layer_id
        is_weight = re.compile(f'^linear{layer_id}_w$')
        for name in self.w.keys():                
            if is_weight.match(name):                
                self.subset[name] = self.w[name]
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
            quantizers[f"linear{layer_id + 1}"] = (self.gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu())

    ## end GPTQ_CUSTOM

    ## Begin SimpleNet_V2
    def my_linear(self, x, weight, bias):
        out = x @ weight.weight + bias
        weight.add_batch(x)
        return out
    ## End SimpleNet_V2


@torch.no_grad()
def quantize_gptq_custom(model, train_loader):
    
    nb_layers = model.nb_layers
    is_last_layer = lambda x: x == (nb_layers - 1)

    nsamples = len(train_loader.dataset)
    batch_size = train_loader.batch_size

    inps = torch.zeros((nsamples, model.N), dtype=torch.float)
    for i, (inp, _) in enumerate(train_loader):
        inps[i*batch_size:(i+1)*batch_size] = inp.view(-1, 32*32)
    outs = torch.zeros_like(inps)

    quantizers = {}

    for layer_id in range(nb_layers):

        if not is_last_layer(layer_id):
    
            model.alloc_gptq(layer_id)

            for i in range(nsamples):
                outs[i] = model.my_linear(inps[i], model.gptq[f"linear{layer_id}_w"], model.w[f"linear{layer_id}_b"])
        
            model.gptq[f"linear{layer_id}_w"].deactivate_add_batch_call = True

            model.fasterquant(layer_id, quantizers)

            for i in range(nsamples):
                outs[i] = model.my_linear(inps[i], model.gptq[f"linear{layer_id}_w"], model.w[f"linear{layer_id}_b"])

            model.free_gptq()

            inps, outs = outs, inps

    return quantizers


def model_pack_custom(model, quantizers, wbits, groupsize):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--eval_gptq", action="store_true")
    parser.add_argument("--train_custom", action="store_true")
    parser.add_argument("--gptq_custom", action="store_true")
    parser.add_argument("--pyquant", action="store_true")

    args = parser.parse_args()

    seed_everything(42)
    lr = 0.02
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.95).load()

    #TODO: Why is training for ref and custom not the same
    #TODO: Custom packing

    ## ================== REFERENCE ==================
    if args.train:
        model = SimpleNet()
        optimizer = optim.Adam(model.parameters(), lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        train(num_epochs, model, optimizer, criterion, train_loader, device)
        torch.save(model.state_dict(), "model.pt")
    elif args.gptq:
        model = SimpleNet()
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt",  map_location="cpu"))
        model = model.to(device)
        quantizers = quantize_gptq(model, train_loader)
        model_pack(model, quantizers, WBITS, GROUPSIZE)
        torch.save(model.state_dict(), "model_quantized.pt") 
        print("Done GPTQ")
    
    elif args.eval_gptq:
        model = SimpleNet()
        device = torch.device("cuda:0")
        model = load_quant(model, "model_quantized.pt", WBITS, GROUPSIZE)
        model = model.to(device)

        start = time.time()
        val_loss, val_acc = evaluate(device, model, criterion, train_loader)
        end = time.time()

        print(f"wbits = {WBITS} using {device}")
        print(f"val_loss: {val_loss:.3f} \t val_acc: {val_acc:.3f}")
        print(f"Latency: {end - start}")
    ## ================== CUSTOM ==================
    elif args.train_custom:
        model = SimpleNet_V2()
        optimizer = optim.Adam(model.parameters(), lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        train(num_epochs, model, optimizer, criterion, train_loader, device)
        torch.save(model.state_dict(), "model_custom.pt")
    elif args.gptq_custom:
        device = torch.device("cpu")
        model = GPTQ_CUSTOM("./model_custom.pt")
        model = model.to(device)
        quantizers = quantize_gptq_custom(model, train_loader)
        model_pack_custom(model, quantizers, WBITS, GROUPSIZE)
        torch.save(model.state_dict(), "model_quantized_custom.pt")
        print("Done Custom GPTQ")
    ## ================== MISC ==================
    elif args.pyquant:
        # Baseline post-training quantization from Pytorch
        model = SimpleNet()
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt"))
        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_prepared = torch.ao.quantization.prepare(model)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_prepared.forward_pyquant(inputs)

        model_quant = torch.ao.quantization.convert(model_prepared)

        start_q = time.time()
        val_loss_q, val_acc_q = evaluate(device, model_quant, criterion, train_loader, is_pyquant=True)
        end_q = time.time()

        print("Pytorch post-training quantization INT8")
        print(model_quant)
        print(f"val_loss_q: {val_loss_q:.3f} \t val_acc_q:{val_acc_q:.3f}")
        print(f"Latency: {end_q - start_q}")
    else:
        # Evaluate float 32
        model = SimpleNet()
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt",  map_location="cpu"))
        model = model.to(device)

        # Evaluate float 32
        start = time.time()
        val_loss, val_acc = evaluate(device, model, criterion, train_loader)
        end = time.time()

        print("Floating point FP32")
        print(f"val_loss: {val_loss:.3f} \t val_acc: {val_acc:.3f}")
        print(f"Latency: {end - start}")
