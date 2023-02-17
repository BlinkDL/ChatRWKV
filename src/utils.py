########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, time, random, os
import copy
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
import yaml
from yaml import CSafeLoader
from inspect import isfunction

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

class TOKENIZER():
    def __init__(self, WORD_NAME, run_device):
        self.tokenizer = Tokenizer.from_file(WORD_NAME)
        self.run_device = run_device

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        return self.tokenizer.encode(x).ids
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, x, ctx_len, temperature=1.0, top_p=1.0):
        probs = F.softmax(logits.float(), dim=-1)

        if self.run_device == "cpu":
            probs = probs.numpy()
            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

class toArgs():
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def save(self, f):
        if type(f) is str:
            f = open(f, 'w')
        yaml.dump(self.get_dict(), f)

    def load(self, f):
        if type(f) is str:
            f = open(f, 'r')
        for k, v in yaml.load(f, CSafeLoader).items():
            self.__setattr__(k, v)

    def get_dict(self):
        _d = {}
        for k, v in self.__dict__.items():
            if '__' not in k or not isfunction(v):
                _d[k] = v
        return _d

    def __str__(self):
        _d = self.get_dict()
        max_len = max([len(k) for k in _d])
        s = [f"{max_len * '='} CONFIGURATION {max_len * '='}"]
        for k, v in _d.items():
            s.append(k.ljust(max_len) + ': ' + str(v))
        return '\n'.join(s)
    
def construct_prompt(f):
    if type(f) is str:
        f = open(f, 'r')
    obj = yaml.load(f, CSafeLoader)
    interface = obj['interface']
    prompt = obj['init_prompt']
    d = {}
    for mode in ['qa', 'chat']:
        d[mode] = {}
        usr_suffix, bot_suffix, intro = 'user', 'bot', 'intro'
        d[mode][usr_suffix] = f"{obj['_'.join([mode, usr_suffix])]}{interface}"
        d[mode][bot_suffix] = f"{obj['_'.join([mode, bot_suffix])]}{interface}"
        _intro = obj['_'.join([mode, intro])] 
        _intro = _intro \
            .replace('{bot}', d[mode][bot_suffix]) \
            .replace('{user}', d[mode][usr_suffix])
        _prompt = copy.deepcopy(prompt)
        d[mode]['init_prompt'] = _prompt \
            .replace('{bot}', d[mode][bot_suffix]) \
            .replace('{user}', d[mode][usr_suffix]) \
            .replace('{interface}', interface) \
            .replace('{intro}', _intro) 
        
    return d
    