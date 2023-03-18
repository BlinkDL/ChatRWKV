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
with open(f"{current_path}/../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

########################################################################################################

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = [187]

########################################################################################################

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
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

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16 *0+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16 *10+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *0+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *10+')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *1 -> cuda fp16')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 -> cpu fp32 *1')
# model = RWKV(model=MODEL_NAME, strategy='cpu fp32')
# model = RWKV(model=MODEL_NAME, strategy='cpu fp32i8')
# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8 *10 -> cuda fp16 *0+')
pipeline = PIPELINE(model, "20B_tokenizer.json")

print('Warmup...')
out, state = model.forward([187, 510, 1563, 310, 247], None, full_output=True)
print(out[-1,:].detach().cpu().numpy())
out, state = model.forward([187], None)
print(out.detach().cpu().numpy())
out, state = model.forward([510, 1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187], None)
out, state = model.forward([510, 1563, 310, 247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187, 510, 1563, 310], None)
out, state = model.forward([247], state)
print(out.detach().cpu().numpy())
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())

########################################################################################################

# init_token = pipeline.encode("In the event that the Purchaser defaults in the payment of any instalment of purchase price, taxes, insurance, interest, or the annual charge described elsewhere herein, or shall default in the performance of any other obligations set forth in this Contract, the Seller may: at his option: (a) Declare immediately due and payable the entire unpaid balance of purchase price, with accrued interest, taxes, and annual charge, and demand full payment thereof, and enforce conveyance of the land by termination of the contract or according to the terms hereof, in which case the Purchaser shall also be liable to the Seller for reasonable attorney's fees for services rendered by any attorney on behalf of the Seller, or (b) sell said land and premises or any part thereof at public auction, in such manner, at such time and place, upon such terms and conditions, and upon such public notice as the Seller may deem best for the interest of all concerned, consisting of advertisement in a newspaper of general circulation in the county or city in which the security property is located at least once a week for Three (3) successive weeks or for such period as applicable law may require and, in case of default of any purchaser, to re-sell with such postponement of sale or resale and upon such public notice thereof as the Seller may determine, and upon compliance by the Purchaser with the terms of sale, and upon judicial approval as may be required by law, convey said land and premises in fee simple to and at the cost of the Purchaser, who shall not be liable to see to the application of the purchase money; and from the proceeds of the sale: First to pay all proper costs and charges, including but not limited to court costs, advertising expenses, auctioneer's allowance, the expenses, if any required to correct any irregularity in the title, premium for Seller's bond, auditor's fee, attorney's fee, and all other expenses of sale occurred in and about the protection and execution of this contract, and all moneys advanced for taxes, assessments, insurance, and with interest thereon as provided herein, and all taxes due upon said land and premises at time of sale, and to retain as compensation a commission of five percent (5%) on the amount of said sale or sales; SECOND, to pay the whole amount then remaining unpaid of the principal of said contract, and interest thereon to date of payment, whether the same shall be due or not, it being understood and agreed that upon such sale before maturity of the contract the balance thereof shall be immediately due and payable; THIRD, to pay liens of record against the security property according to their priority of lien and to the extent that funds remaining in the hands of the Seller are available; and LAST, to pay the remainder of said proceeds, if any, to the vendor, his heirs, personals representatives, successors or assigns upon the delivery and surrender to the vendee of possession of the land and premises, less costs and excess of obtaining possession.")
init_token = pipeline.encode("In the event that the Purchaser defaults in the payment of any instalment of purchase price")

print('Benchmark speed...')
time_slot = {}

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

for i in range(10):
    time_ref = time.time_ns()
    out, state = model.forward(init_token, None)
    aa = out.detach().cpu().numpy()
    record_time('fast')
    print(f"fast {round(time_slot['fast'], 4)}s {aa}")

    time_ref = time.time_ns()
    for j in range(len(init_token)):
        out, state = model.forward([init_token[j]], None if j == 0 else state)
    aa = out.detach().cpu().numpy()
    record_time('slow')
    print(f"slow {round(time_slot['slow'], 4)}s {aa}")

# exit(0)

########################################################################################################

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    logits = 0
    correct = True
    out, model_state = model.forward(src+dst, None, full_output=True)
    for i in range(len(dst)):
        probs = F.softmax(out[len(src)-1+i,:], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
