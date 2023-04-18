import time
import torch
import torch.nn as nn

from rwkv.model import RWKV
from gptq.gptq import *
from gptq.modelutils import *
from gptq.quant import *
from gptq.datautils import *

# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

@torch.no_grad()
def quantize_model(model, train_tokens, device):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # Load layer to device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(device) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(device) 
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    for batch in train_tokens:
        try:
            # model(batch[0].to(device))
            # IndexError: invalid index of a 0-dim tensor.
            #  Use `tensor.item()` in Python or `tensor.item<T>()` 
            # in C++ to convert a 0-dim tensor to a number
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(device)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', type=str, 
        help='Path to model checkpoint file.'
    )

    parser.add_argument(
        '--dataset_name', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )

    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    # ==== DEFAULT ====
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )

    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )

    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )

    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )

    args = parser.parse_args()
    
    # FIXME: Seems like quantization with OPT is not working in CPU mode
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    # Model 
    # model = model = RWKV(args.model_path, strategy='cpu fp32')

    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    
    model.eval()
    
    # Dataset
    train_tokens, test_tokens = get_loaders(
        dataset_name=args.dataset_name,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        model="facebook/opt-125m",
        # model=None
    )
    
    print(f'{len(train_tokens)} train tokens in the text')
    print(f'{len(test_tokens)} test tokens in the text')

    if args.wbits < 16 and not args.nearest:
        start_time = time.time()
        quantizers = quantize_model(model, train_tokens, device)
        end_time = time.time()
        print('Quantization time: ', end_time - start_time)

    if args.save:
        print('Saving quantized model to ', args.save)
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)