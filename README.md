# ChatRWKV
ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model, which is the only RNN (as of now) that can match transformers in quality and scaling, while being faster and saves VRAM.

**Download RWKV-4 weights:** https://huggingface.co/BlinkDL

**RWKV LM:** https://github.com/BlinkDL/RWKV-LM

**RWKV Discord:** https://discord.gg/bDSBUMeFpc

![RWKV-eval](RWKV-eval.png)

It is not instruct-tuned for conversation yet, so don't directly ask it to do stuffs (unless it's a simple question).

Long answer:

```+gen Here is a short story in which Jeff Bezos, Elon Musk, and Bill Gates fight in a tournament:```

```+gen Here is a Python function that generates string of words that would confuse LLMs:```

```
+gen List of penguin facts:

1.
```

```
+gen Q: Can penguins fly?

A: Here is a long answer. Firstly,
```

```
+gen Q: Can penguins fly?

A: Here is some research. Firstly,
```

```
+gen Q: Can penguins fly?

A: Yes, and let me explain why. Firstly,
```

Short answer: 

```+qa Can penguins fly?```

Prompt magic:

```+gen $ curl -i https://google.com/```

```+gen The following is the contents of https://en.wikipedia.org/wiki/Internet:```

```+gen Bob's Blog - Which is better, iOS or Android?```

Chat example from a user:
![ChatRWKV](ChatRWKV.png)

## 中文模型

QQ群 143626394（加入时请简单自我介绍）。

在 chat.py 修改 CHAT_LANG 为 Chinese，修改 MODEL_NAME 为你下载的模型路径。

必须下载双语模型（EngChn），选日期最新的。

大模型：7B 参数，需 14G 显存，效果好（以后可以优化显存占用和速度，但现在忙）：
https://huggingface.co/BlinkDL/rwkv-4-pile-7b/tree/main

中模型：3B 参数，需 6G 显存，效果中等：
https://huggingface.co/BlinkDL/rwkv-4-pile-3b/tree/main

小模型：1.5B 参数 ，需 3G 显存，效果差些：
https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/tree/main

如果没显卡，或者显存不够，可以用 cpu 模式（很慢）。

试试这些指令：
```
+qa 奶茶好喝吗？
+qa 猫喜欢做什么？
+qa How can I learn Python?
+qa 猫会编程吗？
+qa 知乎大V有哪些特点？

+qq 请以《我的驴》为题写一篇作文
+qq 请以《企鹅》为题写一首诗歌

+gen 二向箔是一种超级武器，它的原理是
+gen 我抬头一看，竟然是
+gen 创业思路：\n1.
+gen import torch
```
