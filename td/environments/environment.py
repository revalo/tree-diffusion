from typing import Tuple, Union

from lark import Tree
from td.grammar import Grammar, Compiler
from abc import ABC, abstractclassmethod, abstractproperty, abstractmethod
from functools import lru_cache
import numpy as np

_COMPILER_CACHE_SIZE = 1024


class Environment(ABC):
    @abstractproperty
    def grammar(self) -> Grammar:
        raise NotImplementedError

    @abstractproperty
    def compiler(self) -> Compiler:
        raise NotImplementedError

    @property
    def observation_compiler(self) -> Compiler:
        return self.compiler

    @abstractclassmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def compiled_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def compile_tree(self, expression: Tree):
        return self.compiler.compile(expression)

    def compile_tree_observation(self, expression: Tree):
        return self.observation_compiler.compile(expression)

    @lru_cache(maxsize=_COMPILER_CACHE_SIZE)
    def compile(self, expression: str):
        return self.compile_tree(self.grammar.parse(expression))

    def compile_observation(self, expression: str):
        return self.compile_tree_observation(self.grammar.parse(expression))

    @abstractmethod
    def goal_reached(self, compiledA, compiledB) -> Union[bool, np.ndarray]:
        raise NotImplementedError

    def is_compiled_empty(self, compiled: np.ndarray) -> bool:
        # Check if > 99% of the values are the same.
        flattened = compiled.flatten()
        _, counts = np.unique(flattened, return_counts=True)
        return any(counts / len(flattened) > 0.99)

    def sample_non_empty(self, sample_fn, return_compiled=False, max_attempts=100):
        for _ in range(max_attempts):
            sample = sample_fn()
            compiled = self.compile(sample)
            if not self.is_compiled_empty(compiled):
                if return_compiled:
                    return sample, compiled

                return sample

        if return_compiled:
            return sample, compiled
        return sample
