import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from myRWKV import RWKV
# from rwkv.model import RWKV
import torch.nn as nn

device = "cpu"

# Model
model = RWKV("./RWKV-4-Pile-169M-20220807-8023.pth", strategy='cpu fp32')
# model = RWKV("./1sample_quantized.pth", strategy='cpu fp32')
# Dataset
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
# sentence = "My name is Bob"
# encodings = tokenizer("\n\n".join(sentence), return_tensors='pt')
# ctx_len = 5
# stride = ctx_len // 2
# seq_len = encodings.input_ids.size(1)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
ctx_len = 1024
stride = ctx_len // 2
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
logits, state = None, None
loss_fct = nn.CrossEntropyLoss()

# for begin_loc in tqdm(range(0, seq_len, stride)):
for begin_loc in tqdm(range(0, stride * 3, stride)):
    end_loc = min(begin_loc + ctx_len, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    
    full_logits = torch.zeros((input_ids.size(1), model.w["emb.weight"].shape[0])) 

    with torch.no_grad():
        for i in range(input_ids.size(1)):
            logits, state = model.forward([input_ids[0, i]], state)
            full_logits[i, :] = logits
     
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        labels = target_ids
        labels = labels.to(full_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = full_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

print(f"nlls: {torch.stack(nlls)}")
mean_nll = torch.stack(nlls).mean()
if mean_nll.is_cuda:
    mean_nll = mean_nll.cpu().float()
ppl = torch.exp(mean_nll)
print(f"Perplexity: {ppl}")
