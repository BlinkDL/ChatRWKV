########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1' #### set these before import RWKV
os.environ["RWKV_CUDA_ON"] = '0'
os.environ["RWKV_RESCALE_LAYER"] = '999' # must set this for RWKV-music models and "pip install rwkv --upgrade" to v0.8.12+

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# MODEL_FILE = '/fsx/BlinkDL/HF-MODEL/rwkv-4-music/RWKV-4-MIDI-120M-v1-20230714-ctx4096' # MIDI model
MODEL_FILE = '/fsx/BlinkDL/HF-MODEL/rwkv-5-music/RWKV-5-ABC-82M-v1-20230901-ctx1024' # ABC model (see https://abc.rectanglered.com)

ABC_MODE = ('-ABC-' in MODEL_FILE)
MIDI_MODE = ('-MIDI-' in MODEL_FILE)

model = RWKV(model=MODEL_FILE, strategy='cpu fp32')
pipeline = PIPELINE(model, "tokenizer-midi.json")

if ABC_MODE:
    class ABCTokenizer():
        def __init__(self):
            self.pad_token_id = 0
            self.bos_token_id = 2
            self.eos_token_id = 3
        def encode(self, text):
            ids = [ord(c) for c in text]
            return ids
        def decode(self, ids):
            txt = ''.join(chr(idx) if idx > self.eos_token_id else '' for idx in ids if idx != self.eos_token_id)
            return txt
    tokenizer = ABCTokenizer()
    EOS_ID = tokenizer.eos_token_id
    TOKEN_SEP = ''
elif MIDI_MODE:
    tokenizer = pipeline
    EOS_ID = 0
    TOKEN_SEP = ' '

##########################################################################################################
#
# MIDI model:
# Use https://github.com/briansemrau/MIDI-LLM-tokenizer/blob/main/str_to_midi.py to convert output to MIDI
# Use https://midiplayer.ehubsoft.net/ and select Full MIDI Player (30M) to play MIDI
# For best results: install https://coolsoft.altervista.org/en/virtualmidisynth
# and use soundfont: https://musical-artifacts.com/artifacts/1720
# 
# ABC model:
# Use https://abc.rectanglered.com to play it
# dataset: load_dataset("sander-wood/massive_abcnotation_dataset")["train"]
# Our training data format: [2] + idx + [3], where idx starts with "control code" (https://huggingface.co/datasets/sander-wood/massive_abcnotation_dataset)
#
##########################################################################################################

for TRIAL in range(10):
    print(TRIAL)

    if ABC_MODE:
        ccc_output = 'S:2' # example: 2 segments (can try 1/2/3/4)

        # another example
        ccc_output = '''S:3
B:9
E:4
B:9
E:4
E:4
B:9
L:1/8
M:3/4
K:D
 Bc |"G" d2 cB"A" A2 FE |"Bm" F2 B4 F^G |'''
        ccc = chr(tokenizer.bos_token_id) + ccc_output

        fout = open(f"abc_{TRIAL}.txt", "w")
    elif MIDI_MODE:
        ccc = '<pad>' # the MIDI model is trained using <pad> (0) as separator
        ccc_output = '<start>' # however the str_to_midi.py requires <start> & <end> as separator

        # uncomment this to continue a melody -> style similar to sample2 & sample3
        # ccc = "v:5b:3 v:5b:2 t125 t125 t125 t106 pi:43:5 t24 pi:4a:7 t15 pi:4f:7 t17 pi:56:7 t18 pi:54:7 t125 t49 pi:51:7 t117 pi:4d:7 t125 t125 t111 pi:37:7 t14 pi:3e:6 t15 pi:43:6 t12 pi:4a:7 t17 pi:48:7 t125 t60 pi:45:7 t121 pi:41:7 t125 t117 s:46:5 s:52:5 f:46:5 f:52:5 t121 s:45:5 s:46:0 s:51:5 s:52:0 f:45:5 f:46:0 f:51:5 f:52:0 t121 s:41:5 s:45:0 s:4d:5 s:51:0 f:41:5 f:45:0 f:4d:5 f:51:0 t102 pi:37:0 pi:3e:0 pi:41:0 pi:43:0 pi:45:0 pi:48:0 pi:4a:0 pi:4d:0 pi:4f:0 pi:51:0 pi:54:0 pi:56:0 t19 s:3e:5 s:41:0 s:4a:5 s:4d:0 f:3e:5 f:41:0 f:4a:5 f:4d:0 t121 v:3a:5 t121 v:39:7 t15 v:3a:0 t106 v:35:8 t10 v:39:0 t111 v:30:8 v:35:0 t125 t117 v:32:8 t10 v:30:0 t125 t125 t103 v:5b:0 v:5b:0 t9 pi:4a:7"
        # ccc = '<pad> ' + ccc
        # ccc_output = '<start> pi:4a:7'
        fout = open(f"midi_{TRIAL}.txt", "w")
        
    fout.write(ccc_output)

    occurrence = {}
    state = None
    for i in range(4096): # only trained with ctx4096 (will be longer soon)
        
        if i == 0:
            out, state = model.forward(tokenizer.encode(ccc), state)
        else:
            out, state = model.forward([token], state)

        if MIDI_MODE: # seems only required for MIDI mode
            for n in occurrence:
                out[n] -= (0 + occurrence[n] * 0.5)
        
            out[0] += (i - 2000) / 500 # not too short, not too long
            out[127] -= 1 # avoid "t125"

            # uncomment for piano-only mode
            # out[128:12416] -= 1e10
            # out[13952:20096] -= 1e10
        
        # find the best sampling for your taste
        token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
        # token = pipeline.sample_logits(out, temperature=1.0, top_p=0.7)
        # token = pipeline.sample_logits(out, temperature=1.0, top_p=0.5)

        if token == EOS_ID: break
        
        if MIDI_MODE: # seems only required for MIDI mode
            for n in occurrence: occurrence[n] *= 0.997 #### decay repetition penalty
            if token >= 128 or token == 127:
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            else:
                occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)
        
        fout.write(TOKEN_SEP + tokenizer.decode([token]))
        fout.flush()

    if MIDI_MODE:
        fout.write(' <end>')
    fout.close()
