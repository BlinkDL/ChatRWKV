import numpy as np
import torch
import os
import pathlib
import tokenizers
import random
from myRWKV import RWKV

from datasets import load_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(nsamples, seed, seqlen, model):
    is_rwkv = isinstance(model, RWKV)

    if is_rwkv:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        print('Loading RWKV tokenizer')
        tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '../20B_tokenizer.json'
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        trainenc = torch.unsqueeze(torch.tensor(tokenizer.encode("\n\n".join(traindata['text'])).ids, dtype=torch.long), 0)
        testenc = torch.unsqueeze(torch.tensor(tokenizer.encode("\n\n".join(testdata['text'])).ids, dtype=torch.long), 0)
    else:
        # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        print('Loading tokenizer')
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        # trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        sentence = "My name is Ferdinand and I live in France"
        trainenc = tokenizer("\n\n".join(sentence), return_tensors='pt')
        testenc = tokenizer("\n\n".join(sentence), return_tensors='pt')
    
    random.seed(seed)
    trainloader = []
    shape = trainenc.shape if is_rwkv else trainenc.input_ids.shape
    trainenc = trainenc if is_rwkv else trainenc.input_ids
    random_idx = [random.randint(0, shape[1] - seqlen - 1) for _ in range(nsamples)]

    for i in range(nsamples):
        j = random_idx[i] + seqlen
        inp = trainenc[:, random_idx[i]:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_loaders(
    dataset_name, nsamples, seed, seqlen, model
):
    if 'wikitext2' in dataset_name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in dataset_name:
        raise NotImplementedError('PTB is not supported yet')
        # if 'new' in dataset_name:
        #     return get_ptb_new(nsamples, seed, seqlen, model)
        # return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in dataset_name:
        raise NotImplementedError('C4 is not supported yet')
        # if 'new' in dataset_name:
        #     return get_c4_new(nsamples, seed, seqlen, model)
        # return get_c4(nsamples, seed, seqlen, model)
