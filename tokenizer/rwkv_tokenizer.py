########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, time, random

print('''
#######################################################################################################################

This tokenizer is used in all RWKV Gx and RWKV World models.

Benefits:

* Good support of most languages, from European to CJK to Arabic and Hindi and more.

* Clean vocab. Good for code too. Vocab size = 65536 (use 0 for <|endoftext|>).

* Good at numbers: the numerical tokens are '0'~'9', '10'~'99', ' 0'~' 9', ' 10'~' 99'.

* Very easy tokenization:

** The input text must be in UTF-8.

** Greedy encoding: always pick the longest (in bytes) token (with the highest id) that matches your UTF-8 bytes.

* The tokenization result is surprisingly good, because the vocab respects word boundaries and UTF-8 boundaries.

For 10x faster speed:
mypyc rwkv_tokenizer.py
python3 -c "import rwkv_tokenizer"

#######################################################################################################################
''')

########################################################################################################
# Tokenizer #1 (reference, naive, slow)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

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

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

########################################################################################################
# Tokenizer #2 (trie, faster) https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer
# UPDATE: now much faster
########################################################################################################

class TRIE:
    __slots__ = ("to", "token")
    to:list
    token:int
    def __init__(self):
        self.to = [None for ch in range(256)]
        self.token = 0

    def __repr__(self):
        return "<TRIE token=%s>" % (self.token - 1)

    def add(self, key:bytes, val:int):
        u = self
        for ch in key:
            v = u.to[ch]
            if v is None:
                v = TRIE()
                u.to[ch] = v
            u = v
        u.token = val + 1

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            idx2token[idx] = x

        self.token2idx = {}
        for k,v in idx2token.items():
            self.token2idx[v] = int(k)
        self.idx2token = [b"" for _ in range(max(idx2token) + 1)]
        for idx, token in idx2token.items():
            self.idx2token[idx] = token

        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=i)
        for ch in range(256):
            assert self.root.to[ch] is not None

    def encodeBytes(self, src:bytes):
        tokens = []
        append = tokens.append
        root_to = self.root.to
        idx = 0
        src_len = len(src)
        while idx < src_len:
            u = root_to[src[idx]]
            j = idx + 1
            token = u.token
            end = j
            to = u.to
            while j < src_len:
                u = to[src[j]]
                if u is None:
                    break
                j += 1
                tok = u.token
                if tok:
                    token = tok
                    end = j
                to = u.to
            append(token - 1)
            idx = end
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(self.idx2token.__getitem__, tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

########################################################################################################
# Tokenizer #4 (fast) https://github.com/LoganDark
########################################################################################################

from typing import Generator, Iterable
from ast import literal_eval

# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
FastTokenizerEntry: TypeAlias = Tuple[Optional[int], Dict[int, FastTokenizerEntry]]

class FastTokenizer:
    __slots__ = ('tok2val', 'root')

    tok2val: Dict[int, bytes]
    root: Dict[int, FastTokenizerEntry]

    def __init__(self, file_name) -> None:
        self.tok2val = {}
        self.root = {}

        with open(file_name, 'rt', encoding = 'utf-8') as file:
            for line in file:
                token_str, value_repr = line.rstrip().split(' ', 1)
                value_repr, len_str = value_repr.rsplit(' ', 1)
                value_str: Union[bytes, str] = literal_eval(value_repr)
                value = value_str if isinstance(value_str, bytes) else value_str.encode('utf-8')
                assert len(value) == int(len_str)
                self.add_token(int(token_str), value)

    def add_token(self, token: int, value: bytes) -> None:
        self.tok2val[token] = value
        pos = self.root
        for byte in value[:-1]: pos = pos.setdefault(byte, (None, {}))[1]
        pos.setdefault(value[-1], (token, {}))

    def next_token(self, src: bytes) -> Optional[int]:
        last_token: Optional[int] = None
        last = self.root
        for i in range(0, len(src)):
            if current := last.get(src[i]):
                if token := current[0]: last_token = token
                last = current[1]
            else: break
        return last_token

    def encode_bytes(self, src: bytes) -> Generator[int, None, None]:
        start, stop = 0, len(src)
        while start < stop:
            last_token: Optional[int] = None
            last = self.root

            for i in range(start, stop):
                if current := last.get(src[i]):
                    if token := current[0]:
                        last_token = token
                        start = i + 1
                    last = current[1]
                else: break

            if last_token: yield last_token
            else: break

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        return b''.join(map(self.tok2val.__getitem__, tokens))

    def encode(self, src: str) -> Generator[int, None, None]:
        return self.encode_bytes(src.encode('utf-8'))

    def decode(self, tokens: Iterable[int]) -> str:
        return self.decode_bytes(tokens).decode('utf-8')

########################################################################################################
# Demo
########################################################################################################

TOKENIZER = RWKV_TOKENIZER('rwkv_vocab_v20230424.txt')
TRIE_TEST = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')
FAST_TEST = FastTokenizer('rwkv_vocab_v20230424.txt')

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

########################################################################################################
# Benchmark
########################################################################################################

src = src * 20
src_len = len(src)
print(f'Benchmark {src_len} tokens...')

def benchmark(XXX):
    min_t = 1e100
    for i in range(5):
        t_begin = time.time_ns() - 1
        tokens = list(XXX.encode(src))
        min_t = min(time.time_ns() - t_begin, min_t)
    print('Encode', round(src_len / min_t * 1e3, 3), 'MB/s')

    min_t = 1e100
    for i in range(10):
        t_begin = time.time_ns() - 1
        sss = XXX.decode(tokens)
        min_t = min(time.time_ns() - t_begin, min_t)
    print('Decode', round(src_len / min_t * 1e3, 3), 'MB/s')

benchmark(TOKENIZER)
benchmark(TRIE_TEST)
benchmark(FAST_TEST)

########################################################################################################
# Unit Test
########################################################################################################

print('Unit test...')

QQQ = ['', ' ', 'Õ\U000683b8', b'\xe6\xaa\x81'.decode('utf-8')]

for TRIAL in range(500):
    x = ''
    for xx in [
        ['0',' '],
        ['0','1'],
        ['0','1',' '],
        ['0','1',' ','00','11','  ','000','111','   '],
        list('01 \n\r\t,.;!?:\'\"-=你好')
    ]:
        for i in range(256):
            x += random.choice(xx)
    QQQ += [x]

for i in range(5000):
    QQQ += [' ' * i]

for TRIAL in range(5000):
    x = chr(random.randrange(0, 256))
    x = x * random.randrange(1, 32)
    QQQ += [x]

for TRIAL in range(99999):
    x = chr(random.randrange(256, 1114112))
    x = x * random.randrange(1, 4)
    try:
        tmp = x.encode("utf-8")
        QQQ += [x]
    except:
        pass

QQQ += ['''
UTF-8 decoder capability and stress test
----------------------------------------

Markus Kuhn <http://www.cl.cam.ac.uk/~mgk25/> - 2015-08-28 - CC BY 4.0

This test file can help you examine, how your UTF-8 decoder handles
various types of correct, malformed, or otherwise interesting UTF-8
sequences. This file is not meant to be a conformance test. It does
not prescribe any particular outcome. Therefore, there is no way to
"pass" or "fail" this test file, even though the text does suggest a
preferable decoder behaviour at some places. Its aim is, instead, to
help you think about, and test, the behaviour of your UTF-8 decoder on a
systematic collection of unusual inputs. Experience so far suggests
that most first-time authors of UTF-8 decoders find at least one
serious problem in their decoder using this file.

The test lines below cover boundary conditions, malformed UTF-8
sequences, as well as correctly encoded UTF-8 sequences of Unicode code
points that should never occur in a correct UTF-8 file.

According to ISO 10646-1:2000, sections D.7 and 2.3c, a device
receiving UTF-8 shall interpret a "malformed sequence in the same way
that it interprets a character that is outside the adopted subset" and
"characters that are not within the adopted subset shall be indicated
to the user" by a receiving device. One commonly used approach in
UTF-8 decoders is to replace any malformed UTF-8 sequence by a
replacement character (U+FFFD), which looks a bit like an inverted
question mark, or a similar symbol. It might be a good idea to
visually distinguish a malformed UTF-8 sequence from a correctly
encoded Unicode character that is just not available in the current
font but otherwise fully legal, even though ISO 10646-1 doesn't
mandate this. In any case, just ignoring malformed sequences or
unavailable characters does not conform to ISO 10646, will make
debugging more difficult, and can lead to user confusion.

Please check, whether a malformed UTF-8 sequence is (1) represented at
all, (2) represented by exactly one single replacement character (or
equivalent signal), and (3) the following quotation mark after an
illegal UTF-8 sequence is correctly displayed, i.e. proper
resynchronization takes place immediately after any malformed
sequence. This file says "THE END" in the last line, so if you don't
see that, your decoder crashed somehow before, which should always be
cause for concern.

All lines in this file are exactly 79 characters long (plus the line
feed). In addition, all lines end with "|", except for the two test
lines 2.1.1 and 2.2.1, which contain non-printable ASCII controls
U+0000 and U+007F. If you display this file with a fixed-width font,
these "|" characters should all line up in column 79 (right margin).
This allows you to test quickly, whether your UTF-8 decoder finds the
correct number of characters in every line, that is whether each
malformed sequences is replaced by a single replacement character.

Note that, as an alternative to the notion of malformed sequence used
here, it is also a perfectly acceptable (and in some situations even
preferable) solution to represent each individual byte of a malformed
sequence with a replacement character. If you follow this strategy in
your decoder, then please ignore the "|" column.


Here come the tests:                                                          |
                                                                              |
1  Some correct UTF-8 text                                                    |
                                                                              |
You should see the Greek word 'kosme':       "κόσμε"                          |
                                                                              |
2  Boundary condition test cases                                              |
                                                                              |
2.1  First possible sequence of a certain length                              |
                                                                              |
2.1.1  1 byte  (U-00000000):        "�"                                        
2.1.2  2 bytes (U-00000080):        ""                                       |
2.1.3  3 bytes (U-00000800):        "ࠀ"                                       |
2.1.4  4 bytes (U-00010000):        "𐀀"                                       |
2.1.5  5 bytes (U-00200000):        "�����"                                       |
2.1.6  6 bytes (U-04000000):        "������"                                       |
                                                                              |
2.2  Last possible sequence of a certain length                               |
                                                                              |
2.2.1  1 byte  (U-0000007F):        ""                                        
2.2.2  2 bytes (U-000007FF):        "߿"                                       |
2.2.3  3 bytes (U-0000FFFF):        "￿"                                       |
2.2.4  4 bytes (U-001FFFFF):        "����"                                       |
2.2.5  5 bytes (U-03FFFFFF):        "�����"                                       |
2.2.6  6 bytes (U-7FFFFFFF):        "������"                                       |
                                                                              |
2.3  Other boundary conditions                                                |
                                                                              |
2.3.1  U-0000D7FF = ed 9f bf = "퟿"                                            |
2.3.2  U-0000E000 = ee 80 80 = ""                                            |
2.3.3  U-0000FFFD = ef bf bd = "�"                                            |
2.3.4  U-0010FFFF = f4 8f bf bf = "􏿿"                                         |
2.3.5  U-00110000 = f4 90 80 80 = "����"                                         |
                                                                              |
3  Malformed sequences                                                        |
                                                                              |
3.1  Unexpected continuation bytes                                            |
                                                                              |
Each unexpected continuation byte should be separately signalled as a         |
malformed sequence of its own.                                                |
                                                                              |
3.1.1  First continuation byte 0x80: "�"                                      |
3.1.2  Last  continuation byte 0xbf: "�"                                      |
                                                                              |
3.1.3  2 continuation bytes: "��"                                             |
3.1.4  3 continuation bytes: "���"                                            |
3.1.5  4 continuation bytes: "����"                                           |
3.1.6  5 continuation bytes: "�����"                                          |
3.1.7  6 continuation bytes: "������"                                         |
3.1.8  7 continuation bytes: "�������"                                        |
                                                                              |
3.1.9  Sequence of all 64 possible continuation bytes (0x80-0xbf):            |
                                                                              |
   "����������������                                                          |
    ����������������                                                          |
    ����������������                                                          |
    ����������������"                                                         |
                                                                              |
3.2  Lonely start characters                                                  |
                                                                              |
3.2.1  All 32 first bytes of 2-byte sequences (0xc0-0xdf),                    |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � � � � � � � � �                                           |
    � � � � � � � � � � � � � � � � "                                         |
                                                                              |
3.2.2  All 16 first bytes of 3-byte sequences (0xe0-0xef),                    |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � � � � � � � � � "                                         |
                                                                              |
3.2.3  All 8 first bytes of 4-byte sequences (0xf0-0xf7),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � "                                                         |
                                                                              |
3.2.4  All 4 first bytes of 5-byte sequences (0xf8-0xfb),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � � � "                                                                 |
                                                                              |
3.2.5  All 2 first bytes of 6-byte sequences (0xfc-0xfd),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � "                                                                     |
                                                                              |
3.3  Sequences with last continuation byte missing                            |
                                                                              |
All bytes of an incomplete sequence should be signalled as a single           |
malformed sequence, i.e., you should see only a single replacement            |
character in each of the next 10 tests. (Characters as in section 2)          |
                                                                              |
3.3.1  2-byte sequence with last byte missing (U+0000):     "�"               |
3.3.2  3-byte sequence with last byte missing (U+0000):     "��"               |
3.3.3  4-byte sequence with last byte missing (U+0000):     "���"               |
3.3.4  5-byte sequence with last byte missing (U+0000):     "����"               |
3.3.5  6-byte sequence with last byte missing (U+0000):     "�����"               |
3.3.6  2-byte sequence with last byte missing (U-000007FF): "�"               |
3.3.7  3-byte sequence with last byte missing (U-0000FFFF): "�"               |
3.3.8  4-byte sequence with last byte missing (U-001FFFFF): "���"               |
3.3.9  5-byte sequence with last byte missing (U-03FFFFFF): "����"               |
3.3.10 6-byte sequence with last byte missing (U-7FFFFFFF): "�����"               |
                                                                              |
3.4  Concatenation of incomplete sequences                                    |
                                                                              |
All the 10 sequences of 3.3 concatenated, you should see 10 malformed         |
sequences being signalled:                                                    |
                                                                              |
   "�����������������������������"                                                               |
                                                                              |
3.5  Impossible bytes                                                         |
                                                                              |
The following two bytes cannot appear in a correct UTF-8 string               |
                                                                              |
3.5.1  fe = "�"                                                               |
3.5.2  ff = "�"                                                               |
3.5.3  fe fe ff ff = "����"                                                   |
                                                                              |
4  Overlong sequences                                                         |
                                                                              |
The following sequences are not malformed according to the letter of          |
the Unicode 2.0 standard. However, they are longer then necessary and         |
a correct UTF-8 encoder is not allowed to produce them. A "safe UTF-8         |
decoder" should reject them just like malformed sequences for two             |
reasons: (1) It helps to debug applications if overlong sequences are         |
not treated as valid representations of characters, because this helps        |
to spot problems more quickly. (2) Overlong sequences provide                 |
alternative representations of characters, that could maliciously be          |
used to bypass filters that check only for ASCII characters. For              |
instance, a 2-byte encoded line feed (LF) would not be caught by a            |
line counter that counts only 0x0a bytes, but it would still be               |
processed as a line feed by an unsafe UTF-8 decoder later in the              |
pipeline. From a security point of view, ASCII compatibility of UTF-8         |
sequences means also, that ASCII characters are *only* allowed to be          |
represented by ASCII bytes in the range 0x00-0x7f. To ensure this             |
aspect of ASCII compatibility, use only "safe UTF-8 decoders" that            |
reject overlong UTF-8 sequences for which a shorter encoding exists.          |
                                                                              |
4.1  Examples of an overlong ASCII character                                  |
                                                                              |
With a safe UTF-8 decoder, all of the following five overlong                 |
representations of the ASCII character slash ("/") should be rejected         |
like a malformed UTF-8 sequence, for instance by substituting it with         |
a replacement character. If you see a slash below, you do not have a          |
safe UTF-8 decoder!                                                           |
                                                                              |
4.1.1 U+002F = c0 af             = "��"                                        |
4.1.2 U+002F = e0 80 af          = "���"                                        |
4.1.3 U+002F = f0 80 80 af       = "����"                                        |
4.1.4 U+002F = f8 80 80 80 af    = "�����"                                        |
4.1.5 U+002F = fc 80 80 80 80 af = "������"                                        |
                                                                              |
4.2  Maximum overlong sequences                                               |
                                                                              |
Below you see the highest Unicode value that is still resulting in an         |
overlong sequence if represented with the given number of bytes. This         |
is a boundary test for safe UTF-8 decoders. All five characters should        |
be rejected like malformed UTF-8 sequences.                                   |
                                                                              |
4.2.1  U-0000007F = c1 bf             = "��"                                   |
4.2.2  U-000007FF = e0 9f bf          = "���"                                   |
4.2.3  U-0000FFFF = f0 8f bf bf       = "����"                                   |
4.2.4  U-001FFFFF = f8 87 bf bf bf    = "�����"                                   |
4.2.5  U-03FFFFFF = fc 83 bf bf bf bf = "������"                                   |
                                                                              |
4.3  Overlong representation of the NUL character                             |
                                                                              |
The following five sequences should also be rejected like malformed           |
UTF-8 sequences and should not be treated like the ASCII NUL                  |
character.                                                                    |
                                                                              |
4.3.1  U+0000 = c0 80             = "��"                                       |
4.3.2  U+0000 = e0 80 80          = "���"                                       |
4.3.3  U+0000 = f0 80 80 80       = "����"                                       |
4.3.4  U+0000 = f8 80 80 80 80    = "�����"                                       |
4.3.5  U+0000 = fc 80 80 80 80 80 = "������"                                       |
                                                                              |
5  Illegal code positions                                                     |
                                                                              |
The following UTF-8 sequences should be rejected like malformed               |
sequences, because they never represent valid ISO 10646 characters and        |
a UTF-8 decoder that accepts them might introduce security problems           |
comparable to overlong UTF-8 sequences.                                       |
                                                                              |
5.1 Single UTF-16 surrogates                                                  |
                                                                              |
5.1.1  U+D800 = ed a0 80 = "���"                                                |
5.1.2  U+DB7F = ed ad bf = "���"                                                |
5.1.3  U+DB80 = ed ae 80 = "���"                                                |
5.1.4  U+DBFF = ed af bf = "���"                                                |
5.1.5  U+DC00 = ed b0 80 = "���"                                                |
5.1.6  U+DF80 = ed be 80 = "���"                                                |
5.1.7  U+DFFF = ed bf bf = "���"                                                |
                                                                              |
5.2 Paired UTF-16 surrogates                                                  |
                                                                              |
5.2.1  U+D800 U+DC00 = ed a0 80 ed b0 80 = "������"                               |
5.2.2  U+D800 U+DFFF = ed a0 80 ed bf bf = "������"                               |
5.2.3  U+DB7F U+DC00 = ed ad bf ed b0 80 = "������"                               |
5.2.4  U+DB7F U+DFFF = ed ad bf ed bf bf = "������"                               |
5.2.5  U+DB80 U+DC00 = ed ae 80 ed b0 80 = "������"                               |
5.2.6  U+DB80 U+DFFF = ed ae 80 ed bf bf = "������"                               |
5.2.7  U+DBFF U+DC00 = ed af bf ed b0 80 = "������"                               |
5.2.8  U+DBFF U+DFFF = ed af bf ed bf bf = "������"                               |
                                                                              |
5.3 Noncharacter code positions                                               |
                                                                              |
The following "noncharacters" are "reserved for internal use" by              |
applications, and according to older versions of the Unicode Standard         |
"should never be interchanged". Unicode Corrigendum #9 dropped the            |
latter restriction. Nevertheless, their presence in incoming UTF-8 data       |
can remain a potential security risk, depending on what use is made of        |
these codes subsequently. Examples of such internal use:                      |
                                                                              |
 - Some file APIs with 16-bit characters may use the integer value -1         |
   = U+FFFF to signal an end-of-file (EOF) or error condition.                |
                                                                              |
 - In some UTF-16 receivers, code point U+FFFE might trigger a                |
   byte-swap operation (to convert between UTF-16LE and UTF-16BE).            |
                                                                              |
With such internal use of noncharacters, it may be desirable and safer        |
to block those code points in UTF-8 decoders, as they should never            |
occur legitimately in incoming UTF-8 data, and could trigger unsafe           |
behaviour in subsequent processing.                                           |
                                                                              |
Particularly problematic noncharacters in 16-bit applications:                |
                                                                              |
5.3.1  U+FFFE = ef bf be = "￾"                                                |
5.3.2  U+FFFF = ef bf bf = "￿"                                                |
                                                                              |
Other noncharacters:                                                          |
                                                                              |
5.3.3  U+FDD0 .. U+FDEF = "﷐﷑﷒﷓﷔﷕﷖﷗﷘﷙﷚﷛﷜﷝﷞﷟﷠﷡﷢﷣﷤﷥﷦﷧﷨﷩﷪﷫﷬﷭﷮﷯"|
                                                                              |
5.3.4  U+nFFFE U+nFFFF (for n = 1..10)                                        |
                                                                              |
       "🿾🿿𯿾𯿿𿿾𿿿񏿾񏿿񟿾񟿿񯿾񯿿񿿾񿿿򏿾򏿿                                    |
        򟿾򟿿򯿾򯿿򿿾򿿿󏿾󏿿󟿾󟿿󯿾󯿿󿿾󿿿􏿾􏿿"                                   |
                                                                              |
THE END                                                                       |


UTF-8 encoded sample plain-text file
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

Markus Kuhn [ˈmaʳkʊs kuːn] <http://www.cl.cam.ac.uk/~mgk25/> — 2002-07-25 CC BY


The ASCII compatible UTF-8 encoding used in this plain-text file
is defined in Unicode, ISO 10646-1, and RFC 2279.


Using Unicode/UTF-8, you can write in emails and source code things such as

Mathematics and sciences:

  ∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i),      ⎧⎡⎛┌─────┐⎞⎤⎫
                                            ⎪⎢⎜│a²+b³ ⎟⎥⎪
  ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β),    ⎪⎢⎜│───── ⎟⎥⎪
                                            ⎪⎢⎜⎷ c₈   ⎟⎥⎪
  ℕ ⊆ ℕ₀ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ,                   ⎨⎢⎜       ⎟⎥⎬
                                            ⎪⎢⎜ ∞     ⎟⎥⎪
  ⊥ < a ≠ b ≡ c ≤ d ≪ ⊤ ⇒ (⟦A⟧ ⇔ ⟪B⟫),      ⎪⎢⎜ ⎲     ⎟⎥⎪
                                            ⎪⎢⎜ ⎳aⁱ-bⁱ⎟⎥⎪
  2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm     ⎩⎣⎝i=1    ⎠⎦⎭

Linguistics and dictionaries:

  ði ıntəˈnæʃənəl fəˈnɛtık əsoʊsiˈeıʃn
  Y [ˈʏpsilɔn], Yen [jɛn], Yoga [ˈjoːgɑ]

APL:

  ((V⍳V)=⍳⍴V)/V←,V    ⌷←⍳→⍴∆∇⊃‾⍎⍕⌈

Nicer typography in plain text files:

  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║   • ‘single’ and “double” quotes         ║
  ║                                          ║
  ║   • Curly apostrophes: “We’ve been here” ║
  ║                                          ║
  ║   • Latin-1 apostrophe and accents: '´`  ║
  ║                                          ║
  ║   • ‚deutsche‘ „Anführungszeichen“       ║
  ║                                          ║
  ║   • †, ‡, ‰, •, 3–4, —, −5/+5, ™, …      ║
  ║                                          ║
  ║   • ASCII safety test: 1lI|, 0OD, 8B     ║
  ║                      ╭─────────╮         ║
  ║   • the euro symbol: │ 14.95 € │         ║
  ║                      ╰─────────╯         ║
  ╚══════════════════════════════════════════╝

Combining characters:

  STARGΛ̊TE SG-1, a = v̇ = r̈, a⃑ ⊥ b⃑

Greek (in Polytonic):

  The Greek anthem:

  Σὲ γνωρίζω ἀπὸ τὴν κόψη
  τοῦ σπαθιοῦ τὴν τρομερή,
  σὲ γνωρίζω ἀπὸ τὴν ὄψη
  ποὺ μὲ βία μετράει τὴ γῆ.

  ᾿Απ᾿ τὰ κόκκαλα βγαλμένη
  τῶν ῾Ελλήνων τὰ ἱερά
  καὶ σὰν πρῶτα ἀνδρειωμένη
  χαῖρε, ὦ χαῖρε, ᾿Ελευθεριά!

  From a speech of Demosthenes in the 4th century BC:

  Οὐχὶ ταὐτὰ παρίσταταί μοι γιγνώσκειν, ὦ ἄνδρες ᾿Αθηναῖοι,
  ὅταν τ᾿ εἰς τὰ πράγματα ἀποβλέψω καὶ ὅταν πρὸς τοὺς
  λόγους οὓς ἀκούω· τοὺς μὲν γὰρ λόγους περὶ τοῦ
  τιμωρήσασθαι Φίλιππον ὁρῶ γιγνομένους, τὰ δὲ πράγματ᾿
  εἰς τοῦτο προήκοντα,  ὥσθ᾿ ὅπως μὴ πεισόμεθ᾿ αὐτοὶ
  πρότερον κακῶς σκέψασθαι δέον. οὐδέν οὖν ἄλλο μοι δοκοῦσιν
  οἱ τὰ τοιαῦτα λέγοντες ἢ τὴν ὑπόθεσιν, περὶ ἧς βουλεύεσθαι,
  οὐχὶ τὴν οὖσαν παριστάντες ὑμῖν ἁμαρτάνειν. ἐγὼ δέ, ὅτι μέν
  ποτ᾿ ἐξῆν τῇ πόλει καὶ τὰ αὑτῆς ἔχειν ἀσφαλῶς καὶ Φίλιππον
  τιμωρήσασθαι, καὶ μάλ᾿ ἀκριβῶς οἶδα· ἐπ᾿ ἐμοῦ γάρ, οὐ πάλαι
  γέγονεν ταῦτ᾿ ἀμφότερα· νῦν μέντοι πέπεισμαι τοῦθ᾿ ἱκανὸν
  προλαβεῖν ἡμῖν εἶναι τὴν πρώτην, ὅπως τοὺς συμμάχους
  σώσομεν. ἐὰν γὰρ τοῦτο βεβαίως ὑπάρξῃ, τότε καὶ περὶ τοῦ
  τίνα τιμωρήσεταί τις καὶ ὃν τρόπον ἐξέσται σκοπεῖν· πρὶν δὲ
  τὴν ἀρχὴν ὀρθῶς ὑποθέσθαι, μάταιον ἡγοῦμαι περὶ τῆς
  τελευτῆς ὁντινοῦν ποιεῖσθαι λόγον.

  Δημοσθένους, Γ´ ᾿Ολυνθιακὸς

Georgian:

  From a Unicode conference invitation:

  გთხოვთ ახლავე გაიაროთ რეგისტრაცია Unicode-ის მეათე საერთაშორისო
  კონფერენციაზე დასასწრებად, რომელიც გაიმართება 10-12 მარტს,
  ქ. მაინცში, გერმანიაში. კონფერენცია შეჰკრებს ერთად მსოფლიოს
  ექსპერტებს ისეთ დარგებში როგორიცაა ინტერნეტი და Unicode-ი,
  ინტერნაციონალიზაცია და ლოკალიზაცია, Unicode-ის გამოყენება
  ოპერაციულ სისტემებსა, და გამოყენებით პროგრამებში, შრიფტებში,
  ტექსტების დამუშავებასა და მრავალენოვან კომპიუტერულ სისტემებში.

Russian:

  From a Unicode conference invitation:

  Зарегистрируйтесь сейчас на Десятую Международную Конференцию по
  Unicode, которая состоится 10-12 марта 1997 года в Майнце в Германии.
  Конференция соберет широкий круг экспертов по  вопросам глобального
  Интернета и Unicode, локализации и интернационализации, воплощению и
  применению Unicode в различных операционных системах и программных
  приложениях, шрифтах, верстке и многоязычных компьютерных системах.

Thai (UCS Level 2):

  Excerpt from a poetry on The Romance of The Three Kingdoms (a Chinese
  classic 'San Gua'):

  [----------------------------|------------------------]
    ๏ แผ่นดินฮั่นเสื่อมโทรมแสนสังเวช  พระปกเกศกองบู๊กู้ขึ้นใหม่
  สิบสองกษัตริย์ก่อนหน้าแลถัดไป       สององค์ไซร้โง่เขลาเบาปัญญา
    ทรงนับถือขันทีเป็นที่พึ่ง           บ้านเมืองจึงวิปริตเป็นนักหนา
  โฮจิ๋นเรียกทัพทั่วหัวเมืองมา         หมายจะฆ่ามดชั่วตัวสำคัญ
    เหมือนขับไสไล่เสือจากเคหา      รับหมาป่าเข้ามาเลยอาสัญ
  ฝ่ายอ้องอุ้นยุแยกให้แตกกัน          ใช้สาวนั้นเป็นชนวนชื่นชวนใจ
    พลันลิฉุยกุยกีกลับก่อเหตุ          ช่างอาเพศจริงหนาฟ้าร้องไห้
  ต้องรบราฆ่าฟันจนบรรลัย           ฤๅหาใครค้ำชูกู้บรรลังก์ ฯ

  (The above is a two-column text. If combining characters are handled
  correctly, the lines of the second column should be aligned with the
  | character above.)

Ethiopian:

  Proverbs in the Amharic language:

  ሰማይ አይታረስ ንጉሥ አይከሰስ።
  ብላ ካለኝ እንደአባቴ በቆመጠኝ።
  ጌጥ ያለቤቱ ቁምጥና ነው።
  ደሀ በሕልሙ ቅቤ ባይጠጣ ንጣት በገደለው።
  የአፍ ወለምታ በቅቤ አይታሽም።
  አይጥ በበላ ዳዋ ተመታ።
  ሲተረጉሙ ይደረግሙ።
  ቀስ በቀስ፥ ዕንቁላል በእግሩ ይሄዳል።
  ድር ቢያብር አንበሳ ያስር።
  ሰው እንደቤቱ እንጅ እንደ ጉረቤቱ አይተዳደርም።
  እግዜር የከፈተውን ጉሮሮ ሳይዘጋው አይድርም።
  የጎረቤት ሌባ፥ ቢያዩት ይስቅ ባያዩት ያጠልቅ።
  ሥራ ከመፍታት ልጄን ላፋታት።
  ዓባይ ማደሪያ የለው፥ ግንድ ይዞ ይዞራል።
  የእስላም አገሩ መካ የአሞራ አገሩ ዋርካ።
  ተንጋሎ ቢተፉ ተመልሶ ባፉ።
  ወዳጅህ ማር ቢሆን ጨርስህ አትላሰው።
  እግርህን በፍራሽህ ልክ ዘርጋ።

Runes:

  ᚻᛖ ᚳᚹᚫᚦ ᚦᚫᛏ ᚻᛖ ᛒᚢᛞᛖ ᚩᚾ ᚦᚫᛗ ᛚᚪᚾᛞᛖ ᚾᚩᚱᚦᚹᛖᚪᚱᛞᚢᛗ ᚹᛁᚦ ᚦᚪ ᚹᛖᛥᚫ

  (Old English, which transcribed into Latin reads 'He cwaeth that he
  bude thaem lande northweardum with tha Westsae.' and means 'He said
  that he lived in the northern land near the Western Sea.')

Braille:

  ⡌⠁⠧⠑ ⠼⠁⠒  ⡍⠜⠇⠑⠹⠰⠎ ⡣⠕⠌

  ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠙⠑⠁⠙⠒ ⠞⠕ ⠃⠑⠛⠔ ⠺⠊⠹⠲ ⡹⠻⠑ ⠊⠎ ⠝⠕ ⠙⠳⠃⠞
  ⠱⠁⠞⠑⠧⠻ ⠁⠃⠳⠞ ⠹⠁⠞⠲ ⡹⠑ ⠗⠑⠛⠊⠌⠻ ⠕⠋ ⠙⠊⠎ ⠃⠥⠗⠊⠁⠇ ⠺⠁⠎
  ⠎⠊⠛⠝⠫ ⠃⠹ ⠹⠑ ⠊⠇⠻⠛⠹⠍⠁⠝⠂ ⠹⠑ ⠊⠇⠻⠅⠂ ⠹⠑ ⠥⠝⠙⠻⠞⠁⠅⠻⠂
  ⠁⠝⠙ ⠹⠑ ⠡⠊⠑⠋ ⠍⠳⠗⠝⠻⠲ ⡎⠊⠗⠕⠕⠛⠑ ⠎⠊⠛⠝⠫ ⠊⠞⠲ ⡁⠝⠙
  ⡎⠊⠗⠕⠕⠛⠑⠰⠎ ⠝⠁⠍⠑ ⠺⠁⠎ ⠛⠕⠕⠙ ⠥⠏⠕⠝ ⠰⡡⠁⠝⠛⠑⠂ ⠋⠕⠗ ⠁⠝⠹⠹⠔⠛ ⠙⠑
  ⠡⠕⠎⠑ ⠞⠕ ⠏⠥⠞ ⠙⠊⠎ ⠙⠁⠝⠙ ⠞⠕⠲

  ⡕⠇⠙ ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠁⠎ ⠙⠑⠁⠙ ⠁⠎ ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲

  ⡍⠔⠙⠖ ⡊ ⠙⠕⠝⠰⠞ ⠍⠑⠁⠝ ⠞⠕ ⠎⠁⠹ ⠹⠁⠞ ⡊ ⠅⠝⠪⠂ ⠕⠋ ⠍⠹
  ⠪⠝ ⠅⠝⠪⠇⠫⠛⠑⠂ ⠱⠁⠞ ⠹⠻⠑ ⠊⠎ ⠏⠜⠞⠊⠊⠥⠇⠜⠇⠹ ⠙⠑⠁⠙ ⠁⠃⠳⠞
  ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲ ⡊ ⠍⠊⠣⠞ ⠙⠁⠧⠑ ⠃⠑⠲ ⠔⠊⠇⠔⠫⠂ ⠍⠹⠎⠑⠇⠋⠂ ⠞⠕
  ⠗⠑⠛⠜⠙ ⠁ ⠊⠕⠋⠋⠔⠤⠝⠁⠊⠇ ⠁⠎ ⠹⠑ ⠙⠑⠁⠙⠑⠌ ⠏⠊⠑⠊⠑ ⠕⠋ ⠊⠗⠕⠝⠍⠕⠝⠛⠻⠹
  ⠔ ⠹⠑ ⠞⠗⠁⠙⠑⠲ ⡃⠥⠞ ⠹⠑ ⠺⠊⠎⠙⠕⠍ ⠕⠋ ⠳⠗ ⠁⠝⠊⠑⠌⠕⠗⠎
  ⠊⠎ ⠔ ⠹⠑ ⠎⠊⠍⠊⠇⠑⠆ ⠁⠝⠙ ⠍⠹ ⠥⠝⠙⠁⠇⠇⠪⠫ ⠙⠁⠝⠙⠎
  ⠩⠁⠇⠇ ⠝⠕⠞ ⠙⠊⠌⠥⠗⠃ ⠊⠞⠂ ⠕⠗ ⠹⠑ ⡊⠳⠝⠞⠗⠹⠰⠎ ⠙⠕⠝⠑ ⠋⠕⠗⠲ ⡹⠳
  ⠺⠊⠇⠇ ⠹⠻⠑⠋⠕⠗⠑ ⠏⠻⠍⠊⠞ ⠍⠑ ⠞⠕ ⠗⠑⠏⠑⠁⠞⠂ ⠑⠍⠏⠙⠁⠞⠊⠊⠁⠇⠇⠹⠂ ⠹⠁⠞
  ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠁⠎ ⠙⠑⠁⠙ ⠁⠎ ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲

  (The first couple of paragraphs of "A Christmas Carol" by Dickens)

Compact font selection example text:

  ABCDEFGHIJKLMNOPQRSTUVWXYZ /0123456789
  abcdefghijklmnopqrstuvwxyz £©µÀÆÖÞßéöÿ
  –—‘“”„†•…‰™œŠŸž€ ΑΒΓΔΩαβγδω АБВГДабвгд
  ∀∂∈ℝ∧∪≡∞ ↑↗↨↻⇣ ┐┼╔╘░►☺♀ ﬁ�⑀₂ἠḂӥẄɐː⍎אԱა

Greetings in various languages:

  Hello world, Καλημέρα κόσμε, コンニチハ

Box drawing alignment tests:                                          █
                                                                      ▉
  ╔══╦══╗  ┌──┬──┐  ╭──┬──╮  ╭──┬──╮  ┏━━┳━━┓  ┎┒┏┑   ╷  ╻ ┏┯┓ ┌┰┐    ▊ ╱╲╱╲╳╳╳
  ║┌─╨─┐║  │╔═╧═╗│  │╒═╪═╕│  │╓─╁─╖│  ┃┌─╂─┐┃  ┗╃╄┙  ╶┼╴╺╋╸┠┼┨ ┝╋┥    ▋ ╲╱╲╱╳╳╳
  ║│╲ ╱│║  │║   ║│  ││ │ ││  │║ ┃ ║│  ┃│ ╿ │┃  ┍╅╆┓   ╵  ╹ ┗┷┛ └┸┘    ▌ ╱╲╱╲╳╳╳
  ╠╡ ╳ ╞╣  ├╢   ╟┤  ├┼─┼─┼┤  ├╫─╂─╫┤  ┣┿╾┼╼┿┫  ┕┛┖┚     ┌┄┄┐ ╎ ┏┅┅┓ ┋ ▍ ╲╱╲╱╳╳╳
  ║│╱ ╲│║  │║   ║│  ││ │ ││  │║ ┃ ║│  ┃│ ╽ │┃  ░░▒▒▓▓██ ┊  ┆ ╎ ╏  ┇ ┋ ▎
  ║└─╥─┘║  │╚═╤═╝│  │╘═╪═╛│  │╙─╀─╜│  ┃└─╂─┘┃  ░░▒▒▓▓██ ┊  ┆ ╎ ╏  ┇ ┋ ▏
  ╚══╩══╝  └──┴──┘  ╰──┴──╯  ╰──┴──╯  ┗━━┻━━┛  ▗▄▖▛▀▜   └╌╌┘ ╎ ┗╍╍┛ ┋  ▁▂▃▄▅▆▇█
                                               ▝▀▘▙▄▟

Sanskrit: ﻿काचं शक्नोम्यत्तुम् । नोपहिनस्ति माम् ॥
Sanskrit (standard transcription): kācaṃ śaknomyattum; nopahinasti mām.
Classical Greek: ὕαλον ϕαγεῖν δύναμαι· τοῦτο οὔ με βλάπτει.
Greek (monotonic): Μπορώ να φάω σπασμένα γυαλιά χωρίς να πάθω τίποτα.
Greek (polytonic): Μπορῶ νὰ φάω σπασμένα γυαλιὰ χωρὶς νὰ πάθω τίποτα.
Etruscan: (NEEDED)
Latin: Vitrum edere possum; mihi non nocet.
Old French: Je puis mangier del voirre. Ne me nuit.
French: Je peux manger du verre, ça ne me fait pas mal.
Provençal / Occitan: Pòdi manjar de veire, me nafrariá pas.
Québécois: J'peux manger d'la vitre, ça m'fa pas mal.
Walloon: Dji pou magnî do vêre, çoula m' freut nén må.
Champenois: (NEEDED)
Lorrain: (NEEDED)
Picard: Ch'peux mingi du verre, cha m'foé mie n'ma.
Corsican/Corsu: (NEEDED)
Jèrriais: (NEEDED)
Kreyòl Ayisyen (Haitï): Mwen kap manje vè, li pa blese'm.
Basque: Kristala jan dezaket, ez dit minik ematen.
Catalan / Català: Puc menjar vidre, que no em fa mal.
Spanish: Puedo comer vidrio, no me hace daño.
Aragonés: Puedo minchar beire, no me'n fa mal .
Aranés: (NEEDED)
Mallorquín: (NEEDED)
Galician: Eu podo xantar cristais e non cortarme.
European Portuguese: Posso comer vidro, não me faz mal.
Brazilian Portuguese (8): Posso comer vidro, não me machuca.
Caboverdiano/Kabuverdianu (Cape Verde): M' podê cumê vidru, ca ta maguâ-m'.
Papiamentu: Ami por kome glas anto e no ta hasimi daño.
Italian: Posso mangiare il vetro e non mi fa male.
Milanese: Sôn bôn de magnà el véder, el me fa minga mal.
Roman: Me posso magna' er vetro, e nun me fa male.
Napoletano: M' pozz magna' o'vetr, e nun m' fa mal.
Venetian: Mi posso magnare el vetro, no'l me fa mae.
Zeneise (Genovese): Pòsso mangiâ o veddro e o no me fà mâ.
Sicilian: Puotsu mangiari u vitru, nun mi fa mali.
Campinadese (Sardinia): (NEEDED)
Lugudorese (Sardinia): (NEEDED)
Romansch (Grischun): Jau sai mangiar vaider, senza che quai fa donn a mai.
Romany / Tsigane: (NEEDED)
Romanian: Pot să mănânc sticlă și ea nu mă rănește.
Esperanto: Mi povas manĝi vitron, ĝi ne damaĝas min.
Pictish: (NEEDED)
Breton: (NEEDED)
Cornish: Mý a yl dybry gwéder hag éf ny wra ow ankenya.
Welsh: Dw i'n gallu bwyta gwydr, 'dyw e ddim yn gwneud dolur i mi.
Manx Gaelic: Foddym gee glonney agh cha jean eh gortaghey mee.
Old Irish (Ogham): ᚛᚛ᚉᚑᚅᚔᚉᚉᚔᚋ ᚔᚈᚔ ᚍᚂᚐᚅᚑ ᚅᚔᚋᚌᚓᚅᚐ᚜
Old Irish (Latin): Con·iccim ithi nglano. Ním·géna.
Irish: Is féidir liom gloinne a ithe. Ní dhéanann sí dochar ar bith dom.
Ulster Gaelic: Ithim-sa gloine agus ní miste damh é.
Scottish Gaelic: S urrainn dhomh gloinne ithe; cha ghoirtich i mi.
Anglo-Saxon (Runes): ᛁᚳ᛫ᛗᚨᚷ᛫ᚷᛚᚨᛋ᛫ᛖᚩᛏᚪᚾ᛫ᚩᚾᛞ᛫ᚻᛁᛏ᛫ᚾᛖ᛫ᚻᛖᚪᚱᛗᛁᚪᚧ᛫ᛗᛖ᛬
Anglo-Saxon (Latin): Ic mæg glæs eotan ond hit ne hearmiað me.
Middle English: Ich canne glas eten and hit hirtiþ me nouȝt.
English: I can eat glass and it doesn't hurt me.
English (IPA): [aɪ kæn iːt glɑːs ænd ɪt dɐz nɒt hɜːt miː] (Received Pronunciation)
English (Braille): ⠊⠀⠉⠁⠝⠀⠑⠁⠞⠀⠛⠇⠁⠎⠎⠀⠁⠝⠙⠀⠊⠞⠀⠙⠕⠑⠎⠝⠞⠀⠓⠥⠗⠞⠀⠍⠑
Jamaican: Mi kian niam glas han i neba hot mi.
Lalland Scots / Doric: Ah can eat gless, it disnae hurt us.
Glaswegian: (NEEDED)
Gothic (4): 𐌼𐌰𐌲 𐌲𐌻𐌴𐍃 𐌹̈𐍄𐌰𐌽, 𐌽𐌹 𐌼𐌹𐍃 𐍅𐌿 𐌽𐌳𐌰𐌽 𐌱𐍂𐌹𐌲𐌲𐌹𐌸.
Old Norse (Runes): ᛖᚴ ᚷᛖᛏ ᛖᛏᛁ ᚧ ᚷᛚᛖᚱ ᛘᚾ ᚦᛖᛋᛋ ᚨᚧ ᚡᛖ ᚱᚧᚨ ᛋᚨᚱ
Old Norse (Latin): Ek get etið gler án þess að verða sár.
Norsk / Norwegian (Nynorsk): Eg kan eta glas utan å skada meg.
Norsk / Norwegian (Bokmål): Jeg kan spise glass uten å skade meg.
Føroyskt / Faroese: Eg kann eta glas, skaðaleysur.
Íslenska / Icelandic: Ég get etið gler án þess að meiða mig.
Svenska / Swedish: Jag kan äta glas utan att skada mig.
Dansk / Danish: Jeg kan spise glas, det gør ikke ondt på mig.
Sønderjysk: Æ ka æe glass uhen at det go mæ naue.
Frysk / Frisian: Ik kin glês ite, it docht me net sear.
Nederlands / Dutch: Ik kan glas eten, het doet mĳ geen kwaad.
Kirchröadsj/Bôchesserplat: Iech ken glaas èèse, mer 't deet miech jing pieng.
Afrikaans: Ek kan glas eet, maar dit doen my nie skade nie.
Lëtzebuergescht / Luxemburgish: Ech kan Glas iessen, daat deet mir nët wei.
Deutsch / German: Ich kann Glas essen, ohne mir zu schaden.
Ruhrdeutsch: Ich kann Glas verkasematuckeln, ohne dattet mich wat jucken tut.
Langenfelder Platt: Isch kann Jlaas kimmeln, uuhne datt mich datt weh dääd.
Lausitzer Mundart ("Lusatian"): Ich koann Gloos assn und doas dudd merr ni wii.
Odenwälderisch: Iech konn glaasch voschbachteln ohne dass es mir ebbs daun doun dud.
Sächsisch / Saxon: 'sch kann Glos essn, ohne dass'sch mer wehtue.
Pfälzisch: Isch konn Glass fresse ohne dasses mer ebbes ausmache dud.
Schwäbisch / Swabian: I kå Glas frässa, ond des macht mr nix!
Deutsch (Voralberg): I ka glas eassa, ohne dass mar weh tuat.
Bayrisch / Bavarian: I koh Glos esa, und es duard ma ned wei.
Allemannisch: I kaun Gloos essen, es tuat ma ned weh.
Schwyzerdütsch (Zürich): Ich chan Glaas ässe, das schadt mir nöd.
Schwyzerdütsch (Luzern): Ech cha Glâs ässe, das schadt mer ned.
Plautdietsch: (NEEDED)
Hungarian: Meg tudom enni az üveget, nem lesz tőle bajom.
Suomi / Finnish: Voin syödä lasia, se ei vahingoita minua.
Sami (Northern): Sáhtán borrat lása, dat ii leat bávččas.
Erzian: Мон ярсан суликадо, ды зыян эйстэнзэ а ули.
Northern Karelian: Mie voin syvvä lasie ta minla ei ole kipie.
Southern Karelian: Minä voin syvvä st'oklua dai minule ei ole kibie.
Vepsian: (NEEDED)
Votian: (NEEDED)
Livonian: (NEEDED)
Estonian: Ma võin klaasi süüa, see ei tee mulle midagi.
Latvian: Es varu ēst stiklu, tas man nekaitē.
Lithuanian: Aš galiu valgyti stiklą ir jis manęs nežeidžia
Old Prussian: (NEEDED)
Sorbian (Wendish): (NEEDED)
Czech: Mohu jíst sklo, neublíží mi.
Slovak: Môžem jesť sklo. Nezraní ma.
Polska / Polish: Mogę jeść szkło i mi nie szkodzi.
Slovenian: Lahko jem steklo, ne da bi mi škodovalo.
Bosnian, Croatian, Montenegrin and Serbian (Latin): Ja mogu jesti staklo, i to mi ne šteti.
Bosnian, Montenegrin and Serbian (Cyrillic): Ја могу јести стакло, и то ми не штети.
Macedonian: Можам да јадам стакло, а не ме штета.
Russian: Я могу есть стекло, оно мне не вредит.
Belarusian (Cyrillic): Я магу есці шкло, яно мне не шкодзіць.
Belarusian (Lacinka): Ja mahu jeści škło, jano mne ne škodzić.
Ukrainian: Я можу їсти скло, і воно мені не зашкодить.
Bulgarian: Мога да ям стъкло, то не ми вреди.
Georgian: მინას ვჭამ და არა მტკივა.
Armenian: Կրնամ ապակի ուտել և ինծի անհանգիստ չըներ։
Albanian: Unë mund të ha qelq dhe nuk më gjen gjë.
Turkish: Cam yiyebilirim, bana zararı dokunmaz.
Turkish (Ottoman): جام ييه بلورم بڭا ضررى طوقونمز
Tatar: Алам да бар, пыяла, әмма бу ранит мине.
Uzbek / O’zbekcha: (Roman): Men shisha yeyishim mumkin, ammo u menga zarar keltirmaydi.
Uzbek / Ўзбекча (Cyrillic): Мен шиша ейишим мумкин, аммо у менга зарар келтирмайди.
Bangla / Bengali: আমি কাঁচ খেতে পারি, তাতে আমার কোনো ক্ষতি হয় না।
Marathi (masculine): मी काच खाऊ शकतो, मला ते दुखत नाही.
Marathi (feminine):   मी काच खाऊ शकते, मला ते दुखत नाही.
Kannada: ನನಗೆ ಹಾನಿ ಆಗದೆ, ನಾನು ಗಜನ್ನು ತಿನಬಹುದು
Hindi (masculine): मैं काँच खा सकता हूँ और मुझे उससे कोई चोट नहीं पहुंचती.
Hindi (feminine):   मैं काँच खा सकती हूँ और मुझे उससे कोई चोट नहीं पहुंचती.
Malayalam: എനിക്ക് ഗ്ലാസ് തിന്നാം. അതെന്നെ വേദനിപ്പിക്കില്ല.
Tamil: நான் கண்ணாடி சாப்பிடுவேன், அதனால் எனக்கு ஒரு கேடும் வராது.
Telugu: నేను గాజు తినగలను మరియు అలా చేసినా నాకు ఏమి ఇబ్బంది లేదు
Sinhalese: මට වීදුරු කෑමට හැකියි. එයින් මට කිසි හානියක් සිදු නොවේ.
Urdu(3): میں کانچ کھا سکتا ہوں اور مجھے تکلیف نہیں ہوتی ۔
Pashto(3): زه شيشه خوړلې شم، هغه ما نه خوږوي
Farsi / Persian(3): .من می توانم بدونِ احساس درد شيشه بخورم
Arabic(3): أنا قادر على أكل الزجاج و هذا لا يؤلمني.
Aramaic: (NEEDED)
Maltese: Nista' niekol il-ħġieġ u ma jagħmilli xejn.
Hebrew(3): אני יכול לאכול זכוכית וזה לא מזיק לי.
Yiddish(3): איך קען עסן גלאָז און עס טוט מיר נישט װײ.
Judeo-Arabic: (NEEDED)
Ladino: (NEEDED)
Gǝʼǝz: (NEEDED)
Amharic: (NEEDED)
Twi: Metumi awe tumpan, ɜnyɜ me hwee.
Hausa (Latin): Inā iya taunar gilāshi kuma in gamā lāfiyā.
Hausa (Ajami) (2): إِنا إِىَ تَونَر غِلَاشِ كُمَ إِن غَمَا لَافِىَا
Yoruba(4): Mo lè je̩ dígí, kò ní pa mí lára.
Lingala: Nakokí kolíya biténi bya milungi, ekosála ngáí mabé tɛ́.
(Ki)Swahili: Naweza kula bilauri na sikunyui.
Malay: Saya boleh makan kaca dan ia tidak mencederakan saya.
Tagalog: Kaya kong kumain nang bubog at hindi ako masaktan.
Chamorro: Siña yo' chumocho krestat, ti ha na'lalamen yo'.
Fijian: Au rawa ni kana iloilo, ia au sega ni vakacacani kina.
Javanese: Aku isa mangan beling tanpa lara.
Burmese (Unicode 4.0): က္ယ္ဝန္‌တော္‌၊က္ယ္ဝန္‌မ မ္ယက္‌စားနုိင္‌သည္‌။ ၎က္ရောင္‌့ ထိခုိက္‌မ္ဟု မရ္ဟိပာ။ (9)
Burmese (Unicode 5.0): ကျွန်တော် ကျွန်မ မှန်စားနိုင်တယ်။ ၎င်းကြောင့် ထိခိုက်မှုမရှိပါ။ (9)
Vietnamese (quốc ngữ): Tôi có thể ăn thủy tinh mà không hại gì.
Vietnamese (nôm) (4): 些 𣎏 世 咹 水 晶 𦓡 空 𣎏 害 咦
Khmer: ខ្ញុំអាចញុំកញ្ចក់បាន ដោយគ្មានបញ្ហារ
Lao: ຂອ້ຍກິນແກ້ວໄດ້ໂດຍທີ່ມັນບໍ່ໄດ້ເຮັດໃຫ້ຂອ້ຍເຈັບ.
Thai: ฉันกินกระจกได้ แต่มันไม่ทำให้ฉันเจ็บ
Mongolian (Cyrillic): Би шил идэй чадна, надад хортой биш
Mongolian (Classic) (5): ᠪᠢ ᠰᠢᠯᠢ ᠢᠳᠡᠶᠦ ᠴᠢᠳᠠᠨᠠ ᠂ ᠨᠠᠳᠤᠷ ᠬᠣᠤᠷᠠᠳᠠᠢ ᠪᠢᠰᠢ
Dzongkha: (NEEDED)
Nepali: ﻿म काँच खान सक्छू र मलाई केहि नी हुन्‍न् ।
Tibetan: ཤེལ་སྒོ་ཟ་ནས་ང་ན་གི་མ་རེད།
Chinese: 我能吞下玻璃而不伤身体。
Chinese (Traditional): 我能吞下玻璃而不傷身體。
Taiwanese(6): Góa ē-tàng chia̍h po-lê, mā bē tio̍h-siong.
Japanese: 私はガラスを食べられます。それは私を傷つけません。
Korean: 나는 유리를 먹을 수 있어요. 그래도 아프지 않아요
Bislama: Mi save kakae glas, hemi no save katem mi.
Hawaiian: Hiki iaʻu ke ʻai i ke aniani; ʻaʻole nō lā au e ʻeha.
Marquesan: E koʻana e kai i te karahi, mea ʻā, ʻaʻe hauhau.
Inuktitut (10): ᐊᓕᒍᖅ ᓂᕆᔭᕌᖓᒃᑯ ᓱᕋᙱᑦᑐᓐᓇᖅᑐᖓ
Chinook Jargon: Naika məkmək kakshət labutay, pi weyk ukuk munk-sik nay.
Navajo: Tsésǫʼ yishą́ągo bííníshghah dóó doo shił neezgai da.
Cherokee (and Cree, Chickasaw, Cree, Micmac, Ojibwa, Lakota, Náhuatl, Quechua, Aymara, and other American languages): (NEEDED)
Garifuna: (NEEDED)
Gullah: (NEEDED)
Lojban: mi kakne le nu citka le blaci .iku'i le se go'i na xrani mi
Nórdicg: Ljœr ye caudran créneþ ý jor cẃran.
''']

for q in QQQ:
    tokens = TOKENIZER.encode(q)
    if q != TOKENIZER.decode(tokens):
        print('ERROR', q)
    if str(tokens) != str(TRIE_TEST.encode(q)):
        print('ERROR', q)
    if str(tokens) != str(list(FAST_TEST.encode(q))):
        print('ERROR', q)

print('All OK\n')
