import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sanity_check_utils import seed_everything, MNISTloader, SimpleNet, train, evaluate
from gptq import *
from modelutils import *
from quant import *

WBITS = 8
GROUPSIZE = -1

def quantize_gptq(model, train_loader, device):
    quantizers = {}
    layers = list(model.modules())[1:]
    layers = [l for l in layers if isinstance(l, nn.Linear)]
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
            gptq[name].quantizer.configure(bits=WBITS, perchannel=True, sym=False, mse=False, trits=False)

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

# TODO: perform packing on GPU
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
    make_quant(model, layers, wbits, groupsize)    
    model.load_state_dict(torch.load(checkpoint))
    print('Done.')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--eval_gptq", action="store_true")
    parser.add_argument("--pyquant", action="store_true")

    args = parser.parse_args()

    seed_everything(42)
    lr = 0.02
    num_epochs = 5
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.95).load()

    if args.train:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        train(num_epochs, model, optimizer, criterion, train_loader, device)
        torch.save(model.state_dict(), "model.pt")
    elif args.gptq:
        #FIXME: WHY ON EARTH QUANTIZATION ERROR IS SO DAMN HIGH FOR LAYER 3 AND 4 ?!
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt",  map_location="cpu"))
        model = model.to(device)
        quantizers = quantize_gptq(model, train_loader, device)
        model_pack(model, quantizers, WBITS, GROUPSIZE)
        torch.save(model.state_dict(), "model_quantized.pt") 
        print("Done GPTQ")
    elif args.eval_gptq:
        device = torch.device("cuda:0")
        model = load_quant(model, "model_quantized.pt", WBITS, GROUPSIZE)
        model = model.to(device)

        start = time.time()
        val_loss, val_acc = evaluate(device, model, criterion, train_loader)
        end = time.time()

        print(f"wbits = {WBITS} using {device}")
        print(f"val_loss: {val_loss:.3f} \t val_acc: {val_acc:.3f}")
        print(f"Latency: {end - start}")
    elif args.pyquant:
        # Baseline post-training quantization from Pytorch
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt"))
        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_prepared = torch.ao.quantization.prepare(model)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_prepared.forward(inputs, is_pyquant=True)

        model_quant = torch.ao.quantization.convert(model_prepared)

        start_q = time.time()
        val_loss_q, val_acc_q = evaluate(device, model_quant, criterion, train_loader, is_pyquant=True)
        end_q = time.time()

        print("Pytorch post-training quantization INT8")
        print(model_quant)
        print(f"val_loss_q: {val_loss_q:.3f} \t val_acc_q:{val_acc_q:.3f}")
        print(f"Latency: {end_q - start_q}")
    else:
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
