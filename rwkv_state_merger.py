import torch

base_model = 'E:/RWKV-Runner/models/rwkv-final-v6-2.1-3b.pth'

state_model = 'E:/RWKV-Runner/models/rwkv-x060-chn_single_round_qa-3B-20240505-ctx1024.pth'

out_model = 'E:/RWKV-Runner/models/rwkv-final-v6-2.1-3b-20240505.pth'

print('loading...', base_model)
base = torch.load(base_model, map_location="cpu")

print('loading...', state_model)
state = torch.load(state_model, map_location="cpu")

for k in list(state.keys()):
    base[k] = state[k].clone()

print('saving...', out_model)
torch.save(base, out_model)
