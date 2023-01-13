# ChatRWKV
ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model.

It is not instruct-tuned for conversation yet, so don't expect good quality. But it's already fun.

**Download RWKV-4 weights:** https://huggingface.co/BlinkDL

**RWKV LM:** https://github.com/BlinkDL/RWKV-LM

**RWKV Discord:** https://discord.gg/bDSBUMeFpc

中文用法：在 https://huggingface.co/BlinkDL/rwkv-4-pile-7b/tree/main 下载双语模型（EngChn），在 chat.py 修改 CHAT_LANG 为 Chinese，修改 MODEL_NAME 为你下载的模型路径。目前 7B 模型需要 14G 显存（或者用 cpu 模式，慢很多），可以优化，但是现在忙。QQ群 143626394（加入时请简单自我介绍）。

Chat example from a user:
![ChatRWKV](ChatRWKV.png)
