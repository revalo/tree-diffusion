from abc import ABC, abstractmethod
from typing import Dict, List

from lark import Lark, Tree
from lark.grammar import Terminal
from lark.tree_matcher import TreeMatcher


class Compiler(ABC):
    @abstractmethod
    def compile(
        self,
        expression: Tree,
    ):
        raise NotImplementedError


class Grammar(object):
    def __init__(
        self,
        grammar_spec: str,
        start: str = None,
        sampling_weights: Dict[str, List[float]] = None,
        primitives: List[str] = None,
    ):
        self._grammar_spec = grammar_spec
        self._start_name = start
        self._sampling_weights = sampling_weights or {}
        self._primitives = set(primitives) if primitives else None

        self._lark_parser = Lark(
            grammar_spec,
            start=start,
            propagate_positions=True,
            parser="lalr",
            maybe_placeholders=False,
            lexer="contextual",
        )

        self._tree_matcher = TreeMatcher(self._lark_parser)

        self._initialize_sampler_constants()

        self._lark_parser_for_start = {
            k.value: Lark(
                grammar_spec,
                start=k.value,
                propagate_positions=True,
                parser="lalr",
                maybe_placeholders=False,
                lexer="contextual",
            )
            for k in self._nonterminals.keys()
        }

    def _initialize_sampler_constants(self):
        start = self.lark_parser.options.start
        terminals, rules, ignore_names = self.lark_parser.grammar.compile(start, ())
        ignore_names_set = set(ignore_names)

        names_to_symbols = {}
        for r in rules:
            t = r.origin
            names_to_symbols[t.name] = t

        terminal_map = {}
        for t in terminals:
            names_to_symbols[t.name] = Terminal(t.name)
            if t.name in ignore_names_set:
                continue
            s = t.pattern.value
            terminal_map[t.name] = s

        nonterminals = {}

        for rule in rules:
            nonterminals.setdefault(rule.origin.name, []).append(tuple(rule.expansion))

        allowed_rules = {*terminal_map, *nonterminals}
        while dict(nonterminals) != (
            nonterminals := {
                k: clean
                for k, v in nonterminals.items()
                if (clean := [x for x in v if all(r.name in allowed_rules for r in x)])
            }
        ):
            allowed_rules = {*terminal_map, *nonterminals}

        self._terminal_map = terminal_map
        self._rev_terminal_map = {v: k for k, v in terminal_map.items()}
        self._nonterminals = nonterminals
        self._names_to_symbols = names_to_symbols

        self._vocabulary = sorted(list(set(terminal_map.values())))

        def _compute_min_primitives(x, path=None):
            if path is None:
                path = []
            path_lookup = set(path)

            if x.name in path_lookup:
                return float("inf")

            if x.name in self._primitives:
                return 1

            if isinstance(x, Terminal):
                return 0

            productions = self._nonterminals.get(x.name, [])
            new_path = path + [x.name]

            sub_min_primitives = [
                sum(_compute_min_primitives(x_, new_path) for x_ in p)
                for p in productions
            ]

            rv = min(
                sub_min_primitives,
                default=float("inf"),
            )

            return rv

        all_terminals_and_nonterminals = set()
        for k, v in nonterminals.items():
            for p in v:
                for s in p:
                    all_terminals_and_nonterminals.add(s)

        self._min_primitives = {}
        for x in all_terminals_and_nonterminals:
            self._min_primitives[x] = _compute_min_primitives(x)

        self._min_primitives_choices = {}
        for x in all_terminals_and_nonterminals:
            if isinstance(x, Terminal):
                continue

            choices = self._nonterminals[x.name]
            self._min_primitives_choices[x] = [
                sum(self._min_primitives[y] for y in p) for p in choices
            ]

        self._start_symbol = self._names_to_symbols[self._start_name]

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def vocabulary_map(self):
        return self._terminal_map

    @property
    def rev_vocabulary_map(self):
        return self._rev_terminal_map

    def parse(self, expression: str):
        return self.lark_parser.parse(expression)

    @property
    def lark_parser(self):
        return self._lark_parser

    @property
    def tree_matcher(self):
        return self._tree_matcher

    @property
    def start_symbol(self):
        return self._start_symbol

    @property
    def primitives(self):
        return self._primitives

    @property
    def names_to_symbols(self):
        return self._names_to_symbols

    @property
    def nonterminals(self):
        return self._nonterminals
