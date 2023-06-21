from typing import TYPE_CHECKING, Deque, Dict
from abc import ABCMeta, abstractmethod
from torch import Tensor
from collections import deque


if TYPE_CHECKING:
    from .pipeline import GenerationArgs


class Penalty(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        '''
        Transform the logits with the penalty.
        '''

    @abstractmethod
    def update(self, token: int):
        '''
        Update the penalty with the token.
        '''

    @abstractmethod
    def clear(self):
        '''
        Clear the penalty.
        '''

    @abstractmethod
    def copy(self) -> "Penalty":
        '''
        Copy the penalty, for history.
        '''


class GlobalPenalty(Penalty):
    def __init__(self) -> None:
        self.token_occurrences = {}

    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        for n in self.token_occurrences:
            out[n] -= (args.alpha_presence + self.token_occurrences[n] * args.alpha_frequency)
        return out

    def update(self, token: int):
        if token not in self.token_occurrences:
            self.token_occurrences[token] = 1
        else:
            self.token_occurrences[token] += 1

    def clear(self):
        self.token_occurrences = {}

    def copy(self) -> "GlobalPenalty":
        ret = GlobalPenalty()
        ret.token_occurrences = self.token_occurrences.copy()
        return ret


class SlidingPenalty(Penalty):
    def __init__(self, maxlen: int = 512) -> None:
        self.maxlen = maxlen
        self.token_occurrences: Deque[int] = deque()
        self.occurrences: Dict[int, int] = {}

    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        for n in self.occurrences:
            out[n] -= (args.alpha_presence + self.occurrences[n] * args.alpha_frequency)
        return out

    def update(self, token: int):
        self.token_occurrences.appendleft(token)
        if token not in self.occurrences:
            self.occurrences[token] = 1
        else:
            self.occurrences[token] += 1

        if len(self.token_occurrences) > self.maxlen:
            while len(self.token_occurrences) > self.maxlen:
                token = self.token_occurrences.pop()
                self.occurrences[token] -= 1

    def clear(self):
        self.token_occurrences.clear()
        self.occurrences = {}

    def copy(self) -> "SlidingPenalty":
        ret = SlidingPenalty(self.maxlen)
        ret.token_occurrences = self.token_occurrences.copy()
        ret.occurrences = self.occurrences.copy()
        return ret
