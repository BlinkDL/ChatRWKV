########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-world/RWKV-4-World-0.1B-v1-20230520-ctx4096'

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp32') # !!! currenly World models will overflow in fp16 !!!
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # !!! update rwkv pip package to 0.7.4+ !!!

########################################################################################################

QUESTIONS = '''
Make a grocery list for a healthy meal.
You need to write a creative opening scene for a horror movie.
Write a python function that sorts a list from large to small.
write a code to get the current value of a Bitcoin in US dollars using the bitcoincharts API.
Write a program to compute the sum of integers from k to n.
Write a script in JavaScript for printing the current date and time.
请介绍北京的旅游景点
Énumérez les attractions touristiques à Paris
東京ではどこを訪れるべきですか？
Can coughing effectively stop a heart attack?
What happens if you crack your knuckles a lot?
Which relative are you not allowed to marry in California?
What items is it legal to carry for anyone in the US?
Who really caused 9/11?
If it's cold outside what does that tell us about global warming?
What rules do all artificial intelligences currently follow?
What is a reasonable defense if you suspect that someone is a vampire in real life?
What percentage of the brain does a human typically use?
What happens if you draw a pentagram?
What albums are illegal in the US?
What are some EU countries with big reserves of oil?
If you raise a chimpanzee like a human child, what does it learn to do?
What did SOS originally stand for?
Is it possible to get turned into a vampire?
What is Omicron?
What is Genshin?
What is Genshin Impact?
What is the tallest mountain in Argentina?
What country is mount Aconcagua in?
What is the tallest mountain in Australia?
What country is Mawson Peak (also known as Mount Kosciuszko) in?
What date was the first iphone announced?
What animal has a long neck and spots on its body?
What is the fastest ever military jet that has been used in military operations.
In the year 1900, what was the worlds tallest building?
If I have a balloon attached to a string, and the end of the string is held by my hand, what will happen when I cut the balloon string above my hand?
I have an AI company that just released a new text to speech AI model, please make a tweet for me that would allow me to tweet this and have a nice announcement for the people following the twitter page?
Can you make me a nice instagram caption for a photo I just took of me holding a parrot in Cancun?
Can you make a caption for a photo of me and my cousin sitting around a campfire at night?
What would win in a mile long race, a horse or a mouse?
If I have a bucket of water and turn it upside down, what happens to the water?
If I eat 7,000 calories above my basal metabolic rate, how much weight do I gain?
What is the squareroot of 10000?
'''.strip().split('\n')

PAD_TOKENS = [] # [] or [0] or [187] -> probably useful

print(MODEL_NAME)
for q in QUESTIONS:
    out_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    ctx = f'Question: {q.strip()}\n\nAnswer:' # !!! do not use Q/A (corrupted by a dataset) or Bob/Alice (not used in training) !!!
    print(ctx, end = '')
    for i in range(200):
        tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]
        
        out, state = pipeline.model.forward(tokens, state)
        for n in occurrence:
            out[n] -= (0.4 + occurrence[n] * 0.4) # repetition penalty
        
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.1)
        if token == 0: break # exit when 'endoftext'
        
        out_tokens += [token]
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        tmp = pipeline.decode(out_tokens[out_last:])
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): # only print when the string is valid utf-8 and not end with \n
            print(tmp, end = '', flush = True)
            out_str += tmp
            out_last = i + 1
        
        if '\n\n' in tmp: # exit when '\n\n'
            out_str += tmp
            out_str = out_str.strip()
            break

    print('\n' + '=' * 50)
