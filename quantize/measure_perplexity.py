# Measures perplexity and per-token latency of an RWKV model on a given text file.
# Perplexity is defined here as exp() of average cross-entropy loss.
# Usage: python measure_perplexity.py RWKV-4-Pile-169M-20220807-8023.pth wikitext2 2048

import os
import time
import pathlib
import argparse
import tokenizers
import torch
from typing import List
from rwkv.model import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Measure perplexity and per-token latency of an RWKV model on a given text file')
    parser.add_argument('model_path', help='Path to model checkpoint file')
    parser.add_argument('dataset_path', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('nsamples', help='How many samples', type=int, default=4096)
    return parser.parse_args()

args = parse_args()

def get_wikitext2(nsamples):
    from datasets import load_dataset
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    print('Loading 20B tokenizer (RWKV)')
    tokenizer_path: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

    print('Loading text')
    test_text: str = "\n\n".join(testdata['text'])
    test_tokens = torch.tensor(tokenizer.encode(test_text).ids, dtype=torch.long)
    print(f'{len(test_tokens)} test tokens in the text')

    import random
    random.seed(42)
    # Randomly select a sample of nsamples tokens
    i = random.randint(0, len(test_tokens) - nsamples)    
    return tokenizer, test_tokens[i:i+nsamples]

def get_loaders(dataset_path, nsamples):
    if 'wikitext2' in dataset_path: 
        return get_wikitext2(nsamples)
    else:
        # https://github.com/IST-DASLab/gptq/blob/main/datautils.py
        raise NotImplementedError("Only wikitext2 is supported for now")

tokenizer, test_tokens = get_loaders(args.dataset_path, args.nsamples)

def format_loss(loss: torch.Tensor) -> str:
    return str(['%.3f' % (loss[i].item(),) for i in range(len(loss))]).replace('\'', '')[1:-1]

def format_loss_with_perplexity(loss: torch.Tensor) -> str:
    return f'loss [{format_loss(loss)}], perplexity {"%.3f" % (torch.exp(loss[0]).item(),)}'

# ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

#TODO: Why is PERPLEXITY SO DAMN HIGH ?
model = RWKV(model=args.model_path, strategy='cuda fp16')

logits, state = None, None
loss_sum: torch.Tensor = torch.tensor([0.0], device=device)
loss_count: int = 0
token_count = len(test_tokens)
run_count = token_count - 1
# Ignore 20% of the tokens to let the model warmup
ignore_first_n_tokens = int(token_count * 0.2)
start: float = time.time()

for i in range(run_count):
    token: int = test_tokens[i]
    target: int = test_tokens[i + 1]
        
    logits, state = model.forward([token], None if i == 0 else state)

    if ignore_first_n_tokens == 0 or i + 1 >= ignore_first_n_tokens:
        losses = torch.tensor([
            torch.nn.functional.cross_entropy(logits, torch.tensor(target, dtype=torch.long, device=device), reduction='none').item()
        ]
        , device=device)

        loss_sum += losses
        loss_count += 1

    if i % 100 == 0:
        avg_loss_so_far = loss_sum / loss_count

        duration: float = time.time() - start
        duration_per_token: float = duration / (i + 1)
        runs_remaining: int = run_count - i - 1
        duration_remaining: int = int(runs_remaining * duration_per_token)

        print(f'Token #{i}/{token_count}, '
              f'{int(100.0 * i / token_count)}%, '
              f'ETA {duration_remaining // 60} m {duration_remaining % 60} s', end='')

        if loss_count > 0:
            print(f', averages so far: {format_loss_with_perplexity(avg_loss_so_far)}')
        else:
            print()

print()
print(f'Average latency: {int((time.time() - start) * 1000 / run_count)} ms per token')

print()
print(f'Model: {os.path.basename(args.model_path)}\n'
      f'data: {os.path.basename(args.dataset_path)} with {token_count} tokens\n'
      f'Ignored first {ignore_first_n_tokens} tokens\n'
      f'averages: {format_loss_with_perplexity(loss_sum / loss_count)}')
