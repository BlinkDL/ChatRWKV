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

def quantize_gptq(model, train_loader, device):
    quantizers = {}
    layers = list(model.modules())[1:]
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
            # TODO: 8 bits quantize so that we can compare with pytorch post-training quantization
            gptq[name].quantizer.configure(bits=8, perchannel=True, sym=False, mse=False, trits=False)

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
            gptq[name].fasterquant(percdamp=0.1, groupsize=-1, actorder=False)
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gptq", action="store_true")
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
        device = torch.device("cpu")
        model.load_state_dict(torch.load("./model.pt",  map_location="cpu"))
        model = model.to(device)
        quantize_gptq(model, train_loader, device)
    elif args.pyquant:
        pass
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
