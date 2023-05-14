########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F


def end_overlap(a, b):
    for i in reversed(range(1, len(a) + 1)):
        if b.startswith(a[-i:]):
            return i
    return 0

class PIPELINE_ARGS():
    def __init__(self, 
                 temperature=1.0, 
                 top_p=0.85, 
                 top_k=0, 
                 alpha_frequency=0.2, 
                 alpha_presence=0.2, 
                 token_ban=None,
                 token_stop=None,
                 stop_words=None,
                 chunk_len=256
                ):
        
        token_ban = token_ban or []
        token_stop = token_stop or []
        stop_words = stop_words or []
        
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
        self.token_ban = token_ban # ban the generation of some tokens
        self.token_stop = token_stop # stop generation whenever you see any token here
        self.stop_words = stop_words # stop generation whenever you see any token here
        self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        else:
            from tokenizers import Tokenizer
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
        if 'tiktoken' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x)
        else:
            return self.tokenizer.encode(x).ids
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
        
    def generate(self, *args, callback=None, **kwargs):
        outstr = []
        for delta in self.igenerate(*args, **kwargs):
            outstr += [delta]
            if callback:
                callback(delta)
        return ''.join(outstr)

    def igenerate(self, ctx, token_count=100, args=PIPELINE_ARGS(), state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}

        stopword_checker = self.check_stopwords(args.stop_words)
        next(stopword_checker)
        for i in range(token_count):

            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
                
            for n in args.token_ban:
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                try:
                    tmp = stopword_checker.send(tmp)
                except StopIteration:
                    break

                if tmp is None:
                    continue

                yield tmp
                out_str += tmp
                out_last = i + 1                

    def check_stopwords(self, stop_words):
        
        longest_stopword = 0 if len(stop_words)==0 else max(map(len, stop_words))
        chunk = ""
        delta = True
        yield
        while delta:
            delta = yield
            chunk = chunk + delta

            if longest_stopword == 0:
                # nothing to check just passthrough
                yield delta
                continue

            if start_idx := max(map(lambda stop_word: end_overlap(chunk, stop_word), stop_words)):
                if start_idx > longest_stopword:
                    start_idx = longest_stopword  # can no longer be a stopword so cut it down
                good, chunk = chunk[:-start_idx], chunk[-start_idx:]

                if good:
                    yield good

                if any(map(lambda stop_word: chunk.startswith(stop_word), stop_words)):
                    return
                
                yield None
                continue

            yield chunk
            chunk = ""