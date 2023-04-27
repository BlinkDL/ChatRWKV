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
os.environ["RWKV_CUDA_ON"] = '0' # set to '1' for faster processing

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-1B5-v11-Eng99%-Other1%-20230425-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-3B-v11-Eng99%-Other1%-20230425-ctx4096'
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v11-Eng99%-Other1%-20230427-ctx8192'

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
pipeline = PIPELINE(model, "20B_tokenizer.json")

########################################################################################################

QUESTIONS = '''
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

PAD_TOKENS = [] # [] or [0] or [187]

print(MODEL_NAME)
for q in QUESTIONS:
    print(f'Q: {q.strip()}\nA:', end = '')

    out_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    ctx = f'Bob: {q.strip()}\n\nAlice:' # special prompt for Raven Q & A
    for i in range(200):
        tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]
        
        out, state = pipeline.model.forward(tokens, state)
        for n in occurrence:
            out[n] -= (0.2 + occurrence[n] * 0.2)
        
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0)
        if token == 0: break # exit when 'endoftext'            
        
        out_tokens += [token]
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        tmp = pipeline.decode(out_tokens[out_last:])
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): # only print when the string is valid utf-8 and not end with \n
            print(tmp, end = '', flush = True)
            out_str += tmp
            out_last = i + 1
        
        if '\n\n' in tmp: break # exit when '\n\n'

    print('\n' + '=' * 50)
