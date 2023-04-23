########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys

print('''
#######################################################################################################################

This tokenizer is not used in any RWKV models yet. I plan to use it for the future multilang RWKV models.

Benefits:

* Good support of most languages, from European to CJK to Arabic and Hindi and more.

* Clean vocab. Good for code too. Vocab size = 65525 (use 0 for <|endoftext|>).

* Good at numbers: the numerical tokens are '0'~'9', '10'~'99', ' 0'~' 9', ' 10'~' 99'.

* Very easy tokenization:

** The input text must be in UTF-8.

** Greedy encoding: always pick the longest (in bytes) token (with the highest id) that matches your UTF-8 bytes.

* The tokenization result is surprisingly good, because the vocab respects word boundaries and UTF-8 boundaries.

#######################################################################################################################
''')

class RWKV_TOKENIZER():
    def __init__(self, file_name):
        self.I_TO_TOKEN = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.I_TO_TOKEN[idx] = x

        self.TOKEN_TO_I = {}
        for k,v in self.I_TO_TOKEN.items():
            self.TOKEN_TO_I[v] = int(k)

        # precompute some tables for fast matching

        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src):
        tokens = []
        i = 0
        while True:
            s = src[i:i+1]
            if i < len(src) - 1:
                s0 = int(src[i])
                s1 = int(src[i+1])
                if s1 in self.good[s0]:
                    sss = src[i:i+self.wlen[s0]]
                    for x in self.table[s0][s1]:
                        if sss.startswith(x):
                            s = x
                            break
            tokens += [self.TOKEN_TO_I[s]]
            i += len(s)
            assert i <= len(src)
            if i == len(src):
                break
        return tokens

    def decodeBytes(self, tokens):
        s = b''
        for i in tokens:
            s += self.I_TO_TOKEN[i]
        return s

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.I_TO_TOKEN[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

TOKENIZER = RWKV_TOKENIZER('rwkv_vocab_v20230422.txt')

src = '''起業家イーロン・マスク氏が創業した宇宙開発企業「スペースX（エックス）」の巨大新型ロケット「スターシップ」が20日朝、初めて打ち上げられたが、爆発した。
打ち上げは米テキサス州の東海岸で行われた。無人の試験で、負傷者はいなかった。
打ち上げから2～3分後、史上最大のロケットが制御不能になり、まもなく搭載された装置で破壊された。
マスク氏は、数カ月後に再挑戦すると表明した。
スペースXのエンジニアたちは、それでもこの日のミッションは成功だったとしている。「早期に頻繁に試験する」ことを好む人たちなので、破壊を恐れていない。次のフライトに向け、大量のデータを収集したはずだ。2機目のスターシップは、ほぼ飛行準備が整っている。
マスク氏は、「SpaceXチームの皆さん、スターシップのエキサイティングな試験打ち上げ、おめでとう！　数カ月後に行われる次の試験打ち上げに向けて、多くを学んだ」とツイートした。
アメリカでのロケット打ち上げを認可する米連邦航空局（NASA）は、事故調査を監督するとした。広報担当者は、飛行中に機体が失われた場合の通常の対応だと述べた。
マスク氏は打ち上げ前、期待値を下げようとしていた。発射台の設備を破壊せずに機体を打ち上げるだけでも「成功」だとしていた。
その願いはかなった。スターシップは打ち上げ施設からどんどん上昇し、メキシコ湾の上空へと向かっていった。しかし1分もしないうち、すべてが計画通りに進んでいるのではないことが明らかになった。'''

print(src)
print(f'\n{len(src)} chars')
tokens = TOKENIZER.encode(src)
assert TOKENIZER.decode(tokens) == src
print()
TOKENIZER.printTokens(tokens)
print(f'\n{len(tokens)} tokens\n')
