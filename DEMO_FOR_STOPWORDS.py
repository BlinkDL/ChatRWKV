import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

MODEL_FILE = "/media/yueyulin/KINGSTON/pretrained_models/rwkv/RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth"
model = RWKV(model=MODEL_FILE, strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  #### vocab for rwkv-4-world models
print(model)

print(pipeline)

ctx = "User：请根据以下材料设计一道中餐菜谱。要求生成菜名和具体做法，菜谱最后以”完成！“结束。材料：猪后腿肉，青椒，洋葱，盐，胡椒。\nAssistant：菜名："
print(ctx, end="")

def my_print(s):
    print(s, end="", flush=True)

end_token = pipeline.encode("完成！")
print(end_token)
args = PIPELINE_ARGS(
    temperature=1.5,
    top_p=0.3,
    top_k=0,  # top_k = 0 -> ignore top_k
    alpha_frequency=0.2,  # frequency penalty - see https://platform.openai.com/docs/api-reference/parameter-details
    alpha_presence=0.2,  # presence penalty - see https://platform.openai.com/docs/api-reference/parameter-details
    token_ban=[],  # ban the generation of some tokens
    token_stop=end_token,  # stop generation at these tokens
    chunk_len=256,
)  # split input into chunks to save VRAM (shorter -> less VRAM, but slower)

pipeline.generate(ctx, token_count=1024, args=args, callback=my_print)
print("\n")