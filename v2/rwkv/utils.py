########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        self.tokenizer = Tokenizer.from_file(WORD_NAME)

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

    def sample_logits(self, logits, temperature=1.0, top_p=1.0):
        probs = F.softmax(logits.float(), dim=-1)

        if probs.device == torch.device('cpu'):
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
    
    def generate(self, prompt, max_new_tokens, state=None):
        out = ''
        all_tokens = []
        for i in range(max_new_tokens):
            out, state = self.model.forward(self.encode(prompt) if i == 0 else [token], state)
            token = self.sample_logits(out, temperature=1.0, top_p=0.8)
            all_tokens += [token]
            tmp = self.decode(all_tokens)
            if '\ufffd' not in tmp: # is it a valid utf-8 string?
                out = tmp
        return out
