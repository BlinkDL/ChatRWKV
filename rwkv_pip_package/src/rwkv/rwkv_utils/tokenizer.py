
import os
from os import PathLike
from ..rwkv_tokenizer import TRIE_TOKENIZER
from typing import Generic, List, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    import tiktoken

T = TypeVar('T')


class Tokenizer(Generic[T]):
    def encode(self, x: T) -> List[int]: ...
    def decode(self, tokens: List[int]) -> T: ...
    def validate(self, x: T) -> bool: ...


class Plain(Tokenizer[List[int]]):
    def encode(self, x: List[int]) -> List[int]: return x
    def decode(self, tokens: List[int]) -> List[int]: return tokens
    def validate(self, x: List[int]) -> bool: return True


class StringTokenizer(Tokenizer[str]):
    def validate(self, x: str) -> bool: return '\ufffd' not in x


try:
    from tokenizers import Tokenizer as _Tokenizer

    class HFTokenizer(StringTokenizer):
        def __init__(self, path: PathLike):
            self.tokenizer = _Tokenizer.from_file(path)
            self.encode = self.tokenizer.encode
            self.decode = self.tokenizer.decode

except ImportError:
    ...

try:
    import tiktoken

    class TikTokenizer(StringTokenizer):
        def __init__(self):
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.encode = self.tokenizer.encode
            self.decode = self.tokenizer.decode

except ImportError:
    ...


class RWKVTokenizer(StringTokenizer):
    def __init__(self):
        self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/rwkv_vocab_v20230424.txt')
        self.encode = self.tokenizer.encode
        self.decode = self.tokenizer.decode
