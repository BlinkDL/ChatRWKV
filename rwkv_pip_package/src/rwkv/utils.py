########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from functools import partial
from itertools import chain
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
        
    def infer(self, ctxs, *args, return_logits=False, **kwargs):
        if isinstance(ctxs, str):
            ctxs = [ctxs]
        i, state, out = zip(*self.batch_infer(ctxs, *args, return_logits=True, **kwargs))

        _, order = torch.tensor(i).sort()
        state = [torch.stack(s)[order] for s in state]

        out = torch.stack(out)

        if return_logits:
            return state, out
        return state

    def batch_infer(self, ctxs, args=PIPELINE_ARGS(), state=None, return_logits=False):
        """
        An iterator that yield the tuple (index, state, logits (optional)) of the ctx provided.
        ctx can either be a string or list of strings
        """
        # forward & adjust prob.

        if is_string:=isinstance(ctxs, str):
            ctxs = [ctxs]

        batch_tokens = [self.encode(ctx) for ctx in ctxs]
        idxs = [[(i, j, token) for j, token in enumerate(tokens)] for i, tokens in enumerate(batch_tokens)]
        idxs = list(chain(*idxs))
        x, y, data = zip(*idxs)


        batch_tensor = torch.zeros(max(x)+1, max(y)+1, dtype=torch.int64)
        batch_tensor[x, y] = torch.tensor(data)

        lens = [len(x) for x in batch_tokens]
        out_state = None
        complete = set()

        
        # 1/0
        while batch_tensor.shape[-1] > 0:
            
            chunk_len = min(min([l for l in lens if l > 0]), args.chunk_len)
            
            out, state = self.model.forward(batch_tensor[:, :chunk_len], state)

            # out, state = self.model.forward(batch_tensor[:, :args.chunk_len], state)
            batch_tensor = batch_tensor[:, chunk_len:]

            lens = [l-chunk_len for l in lens]
            
            for i in [i for i, l in enumerate(lens) if l<=0 and i not in complete]:
                
                complete = complete.union({i})

                b_i_state = [s[i] for j, s in enumerate(state)]
                
                return_state = b_i_state if is_string else (i, b_i_state)
                if return_logits:
                    yield *return_state, out[i]
                    continue
                yield return_state
                    
    def generate(self, *args, callback=None, **kwargs):
        outstr = []
        for delta in self.igenerate(*args, **kwargs):
            outstr += [delta]
            if callback:
                callback(delta)
        return ''.join(outstr)

    def igenerate(self, ctx, token_count=100, args=PIPELINE_ARGS(), state=None):
        if is_string := isinstance(ctx, str):
            ctx = [ctx]
        

        batch_size = len(ctx)
        
        i, state, out = zip(*self.batch_infer(ctx, args, state=state, return_logits=True))
        _, sorted_idxs = torch.tensor(i).sort()
        
        
        out = torch.stack(out, axis=0)[sorted_idxs]
        
        state = [torch.stack(s, axis=0)[sorted_idxs] for s in zip(*state)]

        samplers = [self.sampler_state_machine(logits, args, token_count) for logits in out]
        [next(s) for s in samplers]
        PAD = -1 ##??? is there a pad token? should be EOS? 
        def ignore_stop_iter(sampler, logits):
            try:
                return sampler.send(logits)
            except StopIteration as e:
                return (PAD, None)
        tokens = torch.zeros(batch_size, device=out.device).int()
        while not (tokens==PAD).all():
            tokens_tuple, samples = zip(*[ignore_stop_iter(s, logits) for s, logits in zip(samplers, out)])
            tokens = tokens.new(tokens_tuple)
            out, state = self.model.forward(tokens, state)
            # out = out.squeeze(0)
            if is_string:
                
                if samples[0] is not None:
                    yield samples[0]
                continue
            yield samples 
        
    def sampler_state_machine(self, logits, args, token_count):
    
        all_tokens = []
        out_last = 0
        occurrence = {}
        stopword_checker = self.check_stopwords(args.stop_words)
        
        next(stopword_checker)
        yield
        
        
        tmp = None
        for i in range(token_count):
            

            for n in args.token_ban:
                logits[n] = -float('inf')
            for n in occurrence:
                logits[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # sampler
            token = self.sample_logits(logits, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            
            # if len(all_tokens)==1:
            #     tmp = tmp[1:] # strip leading space
            if tmp == '':
                continue
            if '\ufffd' not in tmp: # is valid utf-8 string?

                try:
                    tmp = stopword_checker.send(tmp)
                except StopIteration:
                    break
                out_last = i + 1
            
            
            logits = yield token, tmp

    @staticmethod
    def check_stopwords(stop_words):
        
        longest_stopword = 0 if len(stop_words)==0 else max(map(len, stop_words))
        chunk = ""
        delta = True
        # yield
        to_yield = None
        while delta:
            delta = yield to_yield
            chunk = chunk + delta

            if longest_stopword == 0:
                # nothing to check just passthrough
                to_yield = delta
                continue
            if chunk == '':
                to_yield = None
                continue
            if any(map(lambda stop_word: chunk.startswith(stop_word), stop_words)):
                
                return

            if start_idx := max(map(lambda stop_word: end_overlap(chunk, stop_word), stop_words)):
                if start_idx > longest_stopword:
                    start_idx = longest_stopword  # can no longer be a stopword so cut it down
                good, chunk = chunk[:-start_idx], chunk[-start_idx:]
                if good:
                    to_yield = good
                    continue
                
                to_yield = None
                continue
            
            out = chunk
            chunk = ""
            to_yield = out