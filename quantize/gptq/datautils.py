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
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        print('Loading tokenizer')
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
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

def get_ptb(nsamples, seed, seqlen, model):
    raise NotImplementedError('PTB not implemented yet')
    # from datasets import load_dataset
    # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # from transformers import AutoTokenizer 
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    # testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    # import random
    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    # return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    raise NotImplementedError('C4 not implemented yet')
    # from datasets import load_dataset
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # import random
    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    # import random
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(valdata) - 1)
    #         tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)
    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # valenc = TokenizerWrapper(valenc)

    # return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    raise NotImplementedError('PTB not implemented yet')
    # from datasets import load_dataset
    # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    # testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    # import random
    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    # return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    raise NotImplementedError('C4 not implemented yet')
    # from datasets import load_dataset
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # import random
    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]

    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # valenc = TokenizerWrapper(valenc)

    # return trainloader, valenc


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
