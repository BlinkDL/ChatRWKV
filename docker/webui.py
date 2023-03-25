# modify https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio/blob/main/app.py
import gradio as gr
import os, gc, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1024
title = "RWKV-4-Pile-14B-20230313-ctx8192-test1050"
desc = f'''Links:
<a href='https://github.com/BlinkDL/ChatRWKV' target="_blank" style="margin:0 0.5em">ChatRWKV</a>
<a href='https://github.com/BlinkDL/RWKV-LM' target="_blank" style="margin:0 0.5em">RWKV-LM</a>
<a href="https://pypi.org/project/rwkv/" target="_blank" style="margin:0 0.5em">RWKV pip package</a>
'''

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-14b", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cuda fp16i8 *20 -> cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

def infer(
        ctx,
        token_count=10,
        temperature=1.0,
        top_p=0.8,
        presencePenalty = 0.1,
        countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = []) # stop generation whenever you see any token here

    ctx = ctx.strip(' ')
    if ctx.endswith('\n'):
        ctx = f'\n{ctx.strip()}\n'
    else:
        ctx = f'\n{ctx.strip()}'

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in args.token_ban:
            out[n] = -float('inf')
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()

examples = [
    ["Expert Questions & Helpful Answers\nAsk Research Experts\nQuestion:\nHow can we eliminate poverty?\n\nFull Answer:\n", 150, 1.0, 0.7, 0.2, 0.2],
    ["Here's a short cyberpunk sci-fi adventure story. The story's main character is an artificial human created by a company called OpenBot.\n\nThe Story:\n", 150, 1.0, 0.7, 0.2, 0.2],
    ['''Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Generate a list of adjectives that describe a person as brave.
### Response:
''', 150, 1.0, 0.2, 0.5, 0.5],
    ['''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Arrange the given numbers in ascending order.
### Input:
2, 4, 0, 8, 3
### Response:
''', 150, 1.0, 0.2, 0.5, 0.5],
    ["Ask Expert\n\nQuestion:\nWhat are some good plans for world peace?\n\nExpert Full Answer:\n", 150, 1.0, 0.7, 0.2, 0.2],
    ["Q & A\n\nQuestion:\nWhy is the sky blue?\n\nDetailed Expert Answer:\n", 150, 1.0, 0.7, 0.2, 0.2],
    ["Dear sir,\nI would like to express my boundless apologies for the recent nuclear war.", 150, 1.0, 0.7, 0.2, 0.2],
    ["Here is a shell script to find all .hpp files in /home/workspace and delete the 3th row string of these files:", 150, 1.0, 0.7, 0.1, 0.1],
    ["Building a website can be done in 10 simple steps:\n1.", 150, 1.0, 0.7, 0.2, 0.2],
    ["A Chinese phrase is provided: 百闻不如一见。\nThe masterful Chinese translator flawlessly translates the phrase into English:", 150, 1.0, 0.5, 0.2, 0.2],
    ["I believe the meaning of life is", 150, 1.0, 0.7, 0.2, 0.2],
    ["Simply put, the theory of relativity states that", 150, 1.0, 0.5, 0.2, 0.2],
]


iface = gr.Interface(
    fn=infer,
    description=f'''{desc} *** <b>Please try examples first (bottom of page)</b> *** (edit them to use your question). Demo limited to ctxlen {ctx_limit}.''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=10, label="Prompt", value="Here's a short cyberpunk sci-fi adventure story. The story's main character is an artificial human created by a company called OpenBot.\n\nThe Story:\n"),  # prompt
        gr.Slider(10, 200, step=10, value=150),  # token_count
        gr.Slider(0.2, 2.0, step=0.1, value=1.0),  # temperature
        gr.Slider(0.0, 1.0, step=0.05, value=0.7),  # top_p
        gr.Slider(0.0, 1.0, step=0.1, value=0.2),  # presencePenalty
        gr.Slider(0.0, 1.0, step=0.1, value=0.2),  # countPenalty
    ],
    outputs=gr.Textbox(label="Generated Output", lines=28),
    examples=examples,
    cache_examples=False,
).queue()

demo = gr.TabbedInterface(
    [iface], ["Generative"],
    title=title,
)

demo.queue(max_size=10)
demo.launch(share=False, server_name="0.0.0.0")
