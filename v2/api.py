########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

########################################################################################################
# pip install fastapi uvicron
########################################################################################################

import os, copy, types, gc, sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_path}/../rwkv_pip_package/src")

import numpy as np
from prompt_toolkit import prompt

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

########################################################################################################
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = less accuracy, supports some CPUs
# xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slightly less accuracy
#
# Read https://pypi.org/project/rwkv/ for Strategy Guide
#
########################################################################################################

os.environ[
    "RWKV_JIT_ON"
] = "1"  # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ[
    "RWKV_CUDA_ON"
] = "0"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

MODEL_NAME = "../rwkv/models/RWKV-4-Raven-14B-v8-Eng87%-Chn10%-Jpn1%-Other2%-20230412-ctx4096"

# MODEL_LOAD_STRATEGY = 'cpu fp32'
# MODEL_LOAD_STRATEGY = "cuda fp16"
# MODEL_LOAD_STRATEGY = 'cuda:0 fp16 -> cuda:1 fp16'
# MODEL_LOAD_STRATEGY = 'cuda fp16i8 *10 -> cuda fp16'
# MODEL_LOAD_STRATEGY = 'cuda fp16i8'
# MODEL_LOAD_STRATEGY = 'cuda fp16i8 -> cpu fp32 *10'
MODEL_LOAD_STRATEGY = 'cuda fp16i8 *10+'

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.1  # sometimes it's a good idea to increase temp. try it
GEN_TOP_P = 0.7
GEN_alpha_presence = 0.2  # Presence Penalty
GEN_alpha_frequency = 0.2  # Frequency Penalty
AVOID_REPEAT = "，：？！"

CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# Load Model

print(f"Loading model - {MODEL_NAME} - {MODEL_LOAD_STRATEGY}")
model = RWKV(model=MODEL_NAME, strategy=MODEL_LOAD_STRATEGY)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

def load_prompt(user: str, bot: str):
    init_prompt = f"""
The following is a coherent verbose detailed conversation between a Chinese girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.

{user}: 你好，{bot}。

{bot}: 您好，请问您需要帮助吗？
"""

    init_prompt = init_prompt.strip().split("\n")
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip("\u3000").strip("\r")
    init_prompt = "\n" + ("\n".join(init_prompt)).strip() + "\n\n"
    return init_prompt

########################################################################################################

def run_rnn(model_state, model_tokens, tokens, newline_adj=0):
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj  # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return model_state, model_tokens, out

########################################################################################################

def predict(model_state, model_tokens, prompt):
    prompt = prompt.replace("\\n", "\n").strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P

    model_state, model_tokens, out = run_rnn(
        model_state,
        model_tokens,
        pipeline.encode(f"{prompt}\n\nAlice:"),
        newline_adj=-999999999,
    )

    begin = len(model_tokens)
    out_last = begin
    reply_message = ""
    occurrence = {}
    for i in range(999):
        if i <= 0:
            newline_adj = -999999999
        elif i <= CHAT_LEN_SHORT:
            newline_adj = (i - CHAT_LEN_SHORT) / 10
        elif i <= CHAT_LEN_LONG:
            newline_adj = 0
        else:
            newline_adj = min(
                3, (i - CHAT_LEN_LONG) * 0.25
            )  # MUST END THE GENERATION

        for n in occurrence:
            out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency
        token = pipeline.sample_logits(
            out,
            temperature=x_temp,
            top_p=x_top_p,
        )
        # if token == END_OF_TEXT:
        #     break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        model_state, model_tokens, out = run_rnn(
            model_state, model_tokens, [token], newline_adj=newline_adj
        )
        out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

        xxx = pipeline.decode(model_tokens[out_last:])
        if "\ufffd" not in xxx:  # avoid utf-8 display issues
            reply_message += xxx
            out_last = begin + i + 1

        send_msg = pipeline.decode(model_tokens[begin:])
        if "\n\n" in send_msg:
            send_msg = send_msg.strip()
            break

    reply_message = reply_message.strip()

    return model_state, model_tokens, reply_message

########################################################################################################

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# openai-like chat completions
@app.post("/chat/completions")
async def chat_completions(req: Request):
    model_state = None
    model_tokens = []

    json_data = await req.json()

    init_prompt = load_prompt("Bob", "Alice")

    model_state, model_tokens, message = predict(
        model_state, model_tokens, init_prompt
    )

    prompt = ""
    for message in json_data["messages"]:
        role = "Bob" if message["role"] == "user" else "Alice"
        content = message["content"]
        prompt += f"{role}: {content}\n\n"

    model_state, model_tokens, reply_message = predict(
        model_state, model_tokens, prompt
    )

    json_data["messages"].append(
        {
            "role": "assistant",
            "content": reply_message,
        }
    )

    return json_data

uvicorn.run(app, host="0.0.0.0", port=3000, workers=1)
