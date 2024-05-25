import random

from abc import ABC, abstractmethod
from td.grammar import Grammar
from lark.grammar import Terminal, NonTerminal
from lark import Tree

from dataclasses import dataclass


class GrammarSampler(ABC):
    def __init__(self, grammar: Grammar):
        self._grammar = grammar

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @abstractmethod
    def sample(self, start, **kwargs) -> str:
        raise NotImplementedError


class NaiveRandomSampler(GrammarSampler):
    def sample(self, start) -> str:
        def _sample_inner(current):
            if isinstance(current, Terminal):
                return self.grammar._terminal_map[current.name]

            choices = self.grammar._nonterminals[current.name]
            weights = self.grammar._sampling_weights.get(current.name)

            choice = random.choices(choices, weights=weights)[0]

            return "".join(self.sample(x) for x in choice)

        return _sample_inner(start)


@dataclass
class DerivationChoice:
    partial_expression: str
    unexpanded_start: int
    unexpanded_end: int
    expansion_choices: list
    expansion_index: int
    unexpanded_rule_name: str

    @property
    def pretty(self) -> str:
        rv = self.partial_expression + "\n"
        rv += " " * self.unexpanded_start + "^" * (
            self.unexpanded_end - self.unexpanded_start
        )
        rv += f" -> {self.expansion_choices[self.expansion_index]}"
        return rv


class ConstrainedRandomSampler(GrammarSampler):
    def sample(
        self,
        start,
        min_primitives=4,
        max_primitives=10,
        return_steps=False,
    ):
        num_primitives = random.randint(min_primitives, max_primitives)
        min_primitives = num_primitives
        max_primitives = num_primitives

        assert (
            min_primitives <= max_primitives
        ), "min_primitives must be <= max_primitives"

        tree = Tree(start, [])
        choice_history = []

        def tree_to_string(tree: Tree) -> str:
            if not tree.children:
                if isinstance(tree.data, Terminal):
                    return self.grammar._terminal_map[tree.data.name]
                return ""
            return "".join(tree_to_string(child) for child in tree.children)

        def tree_to_string_node_position(tree: Tree, search_node: Tree):
            def _f(tree: Tree, search_node: Tree, current_start=0) -> tuple[str, int]:
                found = -1

                if tree is search_node:
                    found = current_start

                if not tree.children:
                    if isinstance(tree.data, Terminal):
                        return self.grammar._terminal_map[tree.data.name], found
                    return f"<{tree.data.name}>", found

                current = current_start
                current_rv = ""
                for child in tree.children:
                    stringified, c_found = _f(child, search_node, current)
                    if c_found != -1:
                        found = c_found
                    current_rv += stringified
                    current += len(stringified)

                return current_rv, found

            rv, start = _f(tree, search_node)
            end = start + len(f"<{search_node.data.name}>")
            return rv, start, end

        def pick_expansion(nt, choose_fn=None):
            if return_steps:
                tree_string, start, end = tree_to_string_node_position(tree, nt)

            choices = self.grammar._nonterminals[nt.data.name]
            choice_costs = self.grammar._min_primitives_choices[nt.data]

            if choose_fn is None:
                selected_choices = choices
            else:
                chosen_cost = choose_fn(choice_costs)
                selected_choices = [
                    choice
                    for choice, cost in zip(choices, choice_costs)
                    if cost == chosen_cost
                ]

            chosen = random.choice(selected_choices)

            if return_steps:
                choice_history.append(
                    DerivationChoice(
                        partial_expression=tree_string,
                        unexpanded_start=start,
                        unexpanded_end=end,
                        expansion_choices=choices,
                        expansion_index=choices.index(chosen),
                        unexpanded_rule_name=nt.data.name,
                    )
                )

            return chosen

        def num_primitives_in_tree(tree):
            return sum(num_primitives_in_tree(child) for child in tree.children) + int(
                tree.data.name in self.grammar._primitives
            )

        def get_unexpanded(tree):
            if not len(tree.children):
                if isinstance(tree.data, NonTerminal):
                    return [tree]
                return []

            rv = []
            for child in tree.children:
                rv.extend(get_unexpanded(child))
            return rv

        current_primitives = 0
        unexpanded_min_primitives = self.grammar._min_primitives[start]
        queue = [tree]

        while queue:
            tree_potential = current_primitives + unexpanded_min_primitives

            if tree_potential < min_primitives:
                choice_fn = max
            elif tree_potential > min_primitives and tree_potential < max_primitives:
                choice_fn = None
            else:
                choice_fn = min

            current_unexpanded = random.choice(queue)

            expansion = pick_expansion(current_unexpanded, choice_fn)
            current_primitives += sum(
                item.name in self.grammar._primitives for item in expansion
            )
            unexpanded_min_primitives -= self.grammar._min_primitives[
                current_unexpanded.data
            ]
            unexpanded_min_primitives += sum(
                self.grammar._min_primitives[item] for item in expansion
            )
            current_unexpanded.children = [Tree(item, []) for item in expansion]
            queue.remove(current_unexpanded)
            queue.extend(
                [
                    child
                    for child in current_unexpanded.children
                    if isinstance(child.data, NonTerminal)
                ]
            )

        expression = tree_to_string(tree)

        if return_steps:
            return expression, choice_history

        return expression
