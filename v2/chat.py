########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import argparse
import os, copy, types, gc, sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import logging
import numpy as np
from prompt_toolkit import prompt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

logger = logging.getLogger(__name__)

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

parser = argparse.ArgumentParser()
# Chat setup
parser.add_argument('--chat_lang', type=str, default='English',
                    help='Chat language. One of: English,Chinese. More to come')
parser.add_argument('--model_name', type=str, default=None,
    help='Path to model weights, otherwise will try default. Download RWKV models from https://huggingface.co/BlinkDL'
)
parser.add_argument('--prompt_version', type=str, default=2,
    help="1 for [User & Bot] (Q&A), [Bob & Alice] (chat),"
        "3 for a very long (but great) chat prompt (requires ctx8192, and set RWKV_CUDA_ON = 1 or it will be very slow)"
)
parser.add_argument("--visible_devices", type=str, default=None, help="set CUDA_VISIBLE_DEVICES to value")

# model setup
parser.add_argument('--ctx_len', type=int, default=1024, help='Model context length')
parser.add_argument('--chat_len_short', type=int, default=40, help='')
parser.add_argument('--chat_len_long', type=int, default=150, help='')
parser.add_argument('--free_gen_length', type=int, default=200, help='')
parser.add_argument('--pile_v2_model', type=int, default=0, help='Only for my own testing')

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
parser.add_argument('--gen_temp', type=float, default=1.0,
                    help="Temperature for sampling. Sometimes it's a good idea to increase temp. try it")
parser.add_argument('--gen_top_p', type=float, default=0.8, help='Top-probability for sampling')
parser.add_argument('--gen_alpha_presence', type=float, default=0.2, help='Presence Penalty for sampling')
parser.add_argument('--gen_alpha_frequency', type=float, default=0.2, help='Frequency Penalty for sampling')

# Infra settings
parser.add_argument('--strategy', type=str, default='cuda fp16',
    help='Weight device (cpu,cuda), types (fp32,fp16), quantization (i8), streaming (+), indexing(*10) etc. Examples: '
        'cpu fp32 // cuda fp16 // cuda fp16i8 // cuda fp16i8 *10 -> cuda fp16 // cuda fp16i8 -> cpu fp32 *10 // cuda fp16i8 *10+'
)
parser.add_argument('--rwkv_cuda_on', type=int, default=0,
                    help='1 to compile CUDA kernel (10x faster), requires c++ compiler')
parser.add_argument('--rwkv_jit_on', type=int, default=1, help="'1' or '0', please use torch 1.13+ and benchmark speed")
parser.add_argument('--chunk_len', type=int, default=256,
                    help='Split input into chunks to save VRAM (shorter -> slower)')

args = parser.parse_args()

os.environ['RWKV_CUDA_ON'] = str(args.rwkv_cuda_on)
os.environ['RWKV_JIT_ON'] = str(args.rwkv_jit_on)

# Hardware setup
try:
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
except:
    pass

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if "cuda" in args.strategy and not torch.cuda.is_available():
    logger.warning(f"Could not detect CUDA devices. Setting strategy to CPU.")
    args.strategy = "cpu fp32"

#########################
# Hardware settings
##########################
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


# Use '/' in model path, instead of '\'
# Use convert_model.py to convert a model for a strategy, for faster loading & saves CPU RAM
if args.chat_lang == 'English' and args.model_name is None:
    args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-14B-v6-Eng-20230404-ctx4096'  # try +i for "Alpaca instruct"
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096' # try +i for "Alpaca instruct"
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230313-ctx8192-test1050'
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
    # args.model_name = 'cuda_fp16_RWKV-4-Pile-7B-20230109-ctx4096' # use convert_model.py for faster loading & saves CPU RAM

# testNovel系列是小说模型，请只用 +gen 指令续写。Raven系列可以对话和问答，推荐用 +i 做长问答（只用了小中文语料，纯属娱乐）
elif args.chat_lang == 'Chinese' and args.model_name is None:
    args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v6-ChnEng-20230401-ctx2048' # try +i for "Alpaca instruct"
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-3B-v6-ChnEng-20230401-ctx2048' # try +i for "Alpaca instruct"
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
    # args.model_name = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225'
    # args.model_name = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-663'

# ONLY FOR MY OWN TESTING
args.pile_v2_model = bool(args.pile_v2_model)  # False
# args.model_name = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/v2/3-run1/rwkv-245'
# args.model_name = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/v2/1.5-run1/rwkv-601'
# args.model_name = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/v2/3-run1/rwkv-613'


PROMPT_FILE = f'{current_path}/prompt/default/{args.chat_lang}-{args.prompt_version}.py'
AVOID_REPEAT = '，：？！'


########################################################################################################

print(f'\n{args.chat_lang} - {args.strategy} - {PROMPT_FILE}')
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

print(f'Loading model - {args.model_name}')
model = RWKV(model=args.model_name, strategy=args.strategy)
if not args.pile_v2_model:
    pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
    END_OF_TEXT = 0
    END_OF_LINE = 187
else:
    pipeline = PIPELINE(model, "cl100k_base")
    END_OF_TEXT = 100257
    END_OF_LINE = 198

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd


########################################################################################################

def run_rnn(tokens, newline_adj=0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:args.chunk_len], model_state)
        tokens = tokens[args.chunk_len:]

    out[END_OF_LINE] += newline_adj  # adjust \n probability

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

    msg = message.replace('\\n', '\n').strip()

    x_temp = args.gen_temp
    x_top_p = args.gen_top_p
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
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

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[
                                                                                                :4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            new = (
                "Below is an instruction that describes a task."
                "Write a response that appropriately completes the request.\n"
                "The response should be a single sentence.\n\n"
                f"# Instruction:\n{msg[3:].strip()}\n\n# Response:\n"
            )

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
        for i in range(args.free_gen_len + 100):
            for n in occurrence:
                out[n] -= (args.gen_alpha_presence + occurrence[n] * args.gen_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':  # or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= args.free_gen_len:
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
            elif i <= args.chat_len_short:
                newline_adj = (i - args.chat_len_short) / 10
            elif i <= args.chat_len_long:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - args.chat_len_long) * 0.25)  # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (args.gen_alpha_presence + occurrence[n] * args.gen_alpha_frequency)
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

            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
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

if args.chat_lang == 'English':
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+i YOUR INSTRUCT --> free generation with any instruct. use \\n for new line.
+++ --> continue last free generation (only for +gen / +i)
++ --> retry last free generation (only for +gen / +i)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
'''
elif args.chat_lang == 'Chinese':
    HELP_MSG = f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行，必须用 Raven 模型
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆

+i 某某指令 --> 问独立的问题（忽略上下文），用\\n代表换行，必须用 Raven 模型
+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行，写中文小说必须用 testNovel 模型
+++ --> 继续 +gen / +i 的回答
++ --> 换个 +gen / +i 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文网文 testNovel 模型，可以试这些续写例子（不适合 Raven 模型！）：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.
'''
print(HELP_MSG)
print(f'{args.chat_lang} - {args.model_name} - {args.strategy}')

print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}', f'\n{bot}'), end='')

########################################################################################################

while True:
    msg = prompt(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')
