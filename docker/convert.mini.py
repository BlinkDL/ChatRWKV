from huggingface_hub import hf_hub_download
title = "RWKV-4-Pile-14B-20230313-ctx8192-test1050"
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-14b", filename=f"{title}.pth")

from rwkv.model import RWKV
RWKV(model=model_path, strategy='cuda fp16i8 *0+ -> cpu fp32 *1', convert_and_save_and_exit = f"./models/{title}.pth")
