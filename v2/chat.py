########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import numpy as np
from prompt_toolkit import prompt
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

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
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# Read https://pypi.org/project/rwkv/ for Strategy Guide
#
########################################################################################################

# args.strategy = 'cpu fp32'
args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16'
# args.strategy = 'cuda fp16i8'
# args.strategy = 'cuda fp16i8 -> cpu fp32 *10'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16 *0+'

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '0' # '1' to use CUDA kernel for seq mode (much faster)

CHAT_LANG = 'English' # English // Chinese // more to come

# Download RWKV-4 models from https://huggingface.co/BlinkDL (don't use Instruct-test models unless you use their prompt templates)
# Use '/' in model path, instead of '\'
if CHAT_LANG == 'English':
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230228-ctx4096-test663'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'

elif CHAT_LANG == 'Chinese': # testNovel系列是网文模型，请只用 +gen 指令续写。test4 系列可以问答（只用了小中文语料微调，纯属娱乐）
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-1535-ctx2048-20230306'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/1.5-run1z/rwkv-320'

PROMPT_FILE = f'{current_path}/prompt/default/{CHAT_LANG}-2.py' # -1.py for [User & Bot] (Q&A) prompt, -2.py for [Bob & Alice] (chat) prompt

args.ctx_len = 1024
CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 200

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.0
GEN_TOP_P = 0.85
GEN_alpha_presence = 0.2 # Presence Penalty
GEN_alpha_frequency = 0.2 # Frequency Penalty
AVOID_REPEAT = '，。：？！'

########################################################################################################

print(f'\n{CHAT_LANG} - {args.strategy} - {PROMPT_FILE}')
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

with open(PROMPT_FILE, 'rb') as file:
    user = None
    bot = None
    interface = None
    init_prompt = None
    exec(compile(file.read(), PROMPT_FILE, 'exec'))
init_prompt = init_prompt.strip().split('\n')
for c in range(len(init_prompt)):
    init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'

# Load Model

print(f'Loading model - {args.MODEL_NAME}')
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    out, model_state = model.forward(tokens, model_state)

    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj # adjust \n probability
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################

# Run inference
print(f'\nRun prompt...')

out = run_rnn(pipeline.encode(init_prompt))
save_all_stat('', 'chat_init', out)
gc.collect()
torch.cuda.empty_cache()

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def on_message(message):
    global model_tokens, model_state

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN+100):
            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            occurrence[187] = 0

            if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])
            
            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = pipeline.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = (i - CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            occurrence[187] = 0
            
            out = run_rnn([token], newline_adj=newline_adj)

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = pipeline.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
            
            # send_msg = pipeline.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{pipeline.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

########################################################################################################

if CHAT_LANG == 'English':
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+++ --> continue last free generation (only for +gen / +qa)
++ --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.

Prompt is VERY important. Try all prompts on https://github.com/BlinkDL/ChatRWKV first.
'''
elif CHAT_LANG == 'Chinese':        
    HELP_MSG = f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+qq 某某问题 --> 问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行

注意，中文网文【testNovel】模型，更适合下列指令：
+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+++ --> 继续 +gen / +qa / +qq 的回答
++ --> 换个 +gen / +qa / +qq 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文网文【testNovel】模型，请先试这些续写例子：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.
'''
print(HELP_MSG)
print(f'{CHAT_LANG} - {args.MODEL_NAME} - {args.strategy}')

print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}',f'\n{bot}'), end='')

########################################################################################################

while True:
    msg = prompt(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')
