########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print("RWKV Chat Simple Demo")

import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()

args.strategy = "cuda fp16"  # use CUDA, fp16

args.MODEL_NAME = "E://RWKV-Runner//models//RWKV-5-World-1B5-v2-20231025-ctx4096"

GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.0
GEN_alpha_frequency = 1.0
GEN_penalty_decay = 0.996

CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None


def run_rnn(ctx):
    global model_tokens, model_state

    ctx = ctx.replace("\r\n", "\n")

    tokens = pipeline.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    # print(f"### model ###\n{model_tokens}\n[{pipeline.decode(model_tokens)}]")  # debug

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out


init_ctx = "User: hi" + "\n\n"
init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"

run_rnn(init_ctx)

print(init_ctx, end="")

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    if len(msg) > 0:
        occurrence = {}
        out_tokens = []
        out_last = 0

        out = run_rnn("User: " + msg + "\n\nAssistant:")
        print("\nAssistant:", end="")

        for i in range(99999):
            for n in occurrence:
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
            out[0] -= 1e10  # disable END_OF_TEXT

            token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)

            out, model_state = model.forward([token], model_state)
            model_tokens += [token]

            out_tokens += [token]

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = pipeline.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break
    else:
        print("!!! Error: please say something !!!")
