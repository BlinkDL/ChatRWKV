from collections import deque
from copy import deepcopy
from typing import Any, Deque, Generator, Generic, Optional, TypeVar, List, TYPE_CHECKING
from dataclasses import dataclass, field
import torch
from torch import Tensor
import numpy as np
from .penalty import GlobalPenalty, Penalty, SlidingPenalty
from .tokenizer import RWKVTokenizer, Tokenizer
import torch.nn.functional as F
if TYPE_CHECKING:
    from rwkv.model import RWKV


@dataclass
class GenerationArgs():
    '''
    Data holder for generation arguments.
    '''
    temperature: float = 1.0
    top_p: float = 0.85
    top_k: int = 0
    alpha_frequency: float = 0.2
    alpha_presence: float = 0.2
    token_ban: List[int] = field(default_factory=list)
    token_stop: List[int] = field(default_factory=list)
    chunk_len: int = 256


T = TypeVar('T')


class Pipeline(Generic[T]):

    '''
    A stateless pipeline for RWKV.

    GlobalPenalty is used by default.
    '''

    def __init__(self,
                 model: "RWKV",
                 tokenizer: Tokenizer[T] = RWKVTokenizer(),
                 penalty: Penalty = None,
                 default_args: GenerationArgs = None
                 ) -> None:

        penalty = penalty or GlobalPenalty()
        default_args = default_args or GenerationArgs()

        self.model = model
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.default_args = default_args

        self.encode = tokenizer.encode
        self.decode = tokenizer.decode

    def sample_logits(self, logits, args: GenerationArgs = None):
        args = args or self.default_args

        probs = F.softmax(logits.float(), dim=-1)
        top_k = args.top_k
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def infer(self,
              tokens: List[int], *,
              state: Any = None,
              args: GenerationArgs = None,
              penalty: Penalty = None
              ) -> tuple[Optional[int], Any]:
        '''
        Infer the next token from a list of tokens.

        If the input is a list, and the first element is an integer, it is assumed to be a list of tokens.

        None is returned if stop tokens are generated.
        '''

        args = args or self.default_args
        penalty = penalty or self.penalty

        for token in tokens:
            penalty.update(token)

        for i in range(0, len(tokens), args.chunk_len):
            chunk = tokens[i:i + args.chunk_len]
            out, state = self.model.forward(chunk, state=state)

        for n in args.token_ban:
            out[n] = -float('inf')

        out = penalty.transform(out, args)
        token = self.sample_logits(out, args=args)
        if token in args.token_stop:
            return None, state

        penalty.update(token)
        return token, state

    def generate(self,
                 ctx: T,
                 generation_length: int = 100, *,
                 state=None,
                 args: GenerationArgs = None,
                 clear_penalty: bool = True
                 ) -> Generator[T, None, None]:
        if args is None:
            args = self.default_args

        if clear_penalty:
            self.penalty.clear()

        tokens_tmp = []
        token, state = self.infer(self.encode(ctx), state=state, args=args)
        while token is not None and generation_length > 0:
            generation_length -= 1
            tokens_tmp.append(token)
            tmp = self.decode(tokens_tmp)
            if self.tokenizer.validate(tmp):
                yield tmp
                tokens_tmp = []
            token, state = self.infer([token], state=state, args=args)


class StatefulPipeline(Generic[T]):

    '''
    A stateful pipeline for RWKV.

    The pipeline holds the state that can act as 'memory' for the model.

    SlidingPenalty with maxlen=1024 is used by default.
    '''

    state: List[Tensor]
    last_token: Optional[int]

    def __init__(
        self,
        model: "RWKV",
        tokenizer: Tokenizer[T] = RWKVTokenizer(),
        penalty: Penalty = None,
        default_args: GenerationArgs = None,
        initial_state: List[Tensor] = None,
        initial_prompt: T = None
    ) -> None:
        if initial_prompt is not None and initial_state is not None:
            raise ValueError('Cannot provide both initial_state and initial_prompt')

        penalty = penalty or SlidingPenalty()
        default_args = default_args or GenerationArgs()

        self.model = model
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.default_args = default_args

        self.encode = tokenizer.encode
        self.decode = tokenizer.decode

        self.state = initial_state
        self.last_token = None
        if initial_prompt is not None:
            self.push(initial_prompt)

    def push(self, ctx: T, args: GenerationArgs = None) -> None:
        '''
        Push a context into the state.

        Last token is inferred from the context, and will be used as the first token for the completion.
        '''
        args = args or self.default_args
        self.last_token, self.state = self.infer(self.encode(ctx), args=args)

    def sample_logits(self, logits, args: GenerationArgs = None):
        args = args or self.default_args

        probs = F.softmax(logits.float(), dim=-1)
        top_k = args.top_k
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def infer(
        self,
        tokens: List[int],
        args: GenerationArgs = None
    ) -> tuple[Optional[int], Any]:
        '''
        Generate exactly one token from the given list of token.

        `None` is returned if the token is a stop token.

        The state is updated after the generation.

        If `state` is `None`, the internal state is used, and the internal state is updated.
        '''
        args = args or self.default_args

        for token in tokens:
            self.penalty.update(token)

        for i in range(0, len(tokens), args.chunk_len):
            chunk = tokens[i:i + args.chunk_len]
            out, self.state = self.model.forward(chunk, state=self.state)

        for n in args.token_ban:
            out[n] = -float('inf')

        out = self.penalty.transform(out, args)
        token = self.sample_logits(out, args=args)
        if token in args.token_stop:
            return None, self.state

        self.penalty.update(token)
        self.last_token = token
        return token, self.state

    def _generate(
            self,
            ctx: List[int],
            generation_length: int = 100, *,
            args: GenerationArgs = None
    ) -> Generator[List[int], None, None]:
        args = args or self.default_args

        tokens_tmp = []
        token, self.state = self.infer(ctx, args=args)
        while token is not None and generation_length > 0:
            generation_length -= 1
            tokens_tmp.append(token)
            tmp = self.decode(tokens_tmp)
            if self.tokenizer.validate(tmp):
                yield tmp
                tokens_tmp = []
            token, self.state = self.infer([token], args=args)

    def generate(
            self,
            ctx: T,
            generation_length: int = 100, *,
            args: GenerationArgs = None
    ) -> Generator[T, None, None]:
        '''
        Generates encoded tokens from the given context.

        The return value is a generator that yields the generated parts of the string.

        The state is updated as generation.
        '''
        return self._generate(self.encode(ctx), generation_length=generation_length, args=args)

    def continue_generation(self, generation_length: int = 100, *, args: GenerationArgs = None) -> Generator[T, None, None]:
        '''
        Continue the generation from the last token.

        The return value is a generator that yields the generated parts of the string.

        The state is updated as generation.
        '''
        return self._generate([self.last_token], generation_length=generation_length, args=args)


class RecallablePipeline(StatefulPipeline[T]):
    '''
    A stateful pipeline that can recall the last generations.

    However, only generation is supported, infer is not supported as memory would explode.
    '''

    history: Deque[List[Tensor]]
    history_tokens: Deque[int]
    history_penalties: Deque[Penalty]

    def __init__(self,
                 model: "RWKV",
                 tokenizer: Tokenizer = RWKVTokenizer(),
                 penalty: Penalty = None,
                 default_args: GenerationArgs = None,
                 initial_state: List[Tensor] = None,
                 initial_prompt: T = None,
                 max_history: int = 16
                 ) -> None:
        self.history = deque(maxlen=max_history)
        self.history_tokens = deque(maxlen=max_history)
        self.history_penalties = deque(maxlen=max_history)
        super().__init__(model, tokenizer, penalty, default_args, initial_state, initial_prompt)

    def recall(self, times: int = 1) -> None:
        '''
        Recall the last generation.
        '''
        for _ in range(times):
            if len(self.history) == 0:
                raise IndexError('Cannot recall empty history')
            self.state = self.history.pop()
            self.last_token = self.history_tokens.pop()
            self.penalty = self.history_penalties.pop()

    def push(self, ctx: T) -> None:
        '''
        Push the current state to the history.
        '''
        if self.state is not None:
            self.history.append(deepcopy(self.state))
            self.history_tokens.append(self.last_token)
            self.history_penalties.append(self.penalty.copy())
        return super().push(ctx)

    def _generate(self, ctx: List[int],
                  generation_length: int = 100, *, args: GenerationArgs = None) -> Generator[Any, None, None]:
        self.history.append(deepcopy(self.state))
        self.history_tokens.append(self.last_token)
        self.history_penalties.append(self.penalty.copy())
        return super()._generate(ctx, generation_length, args=args)
