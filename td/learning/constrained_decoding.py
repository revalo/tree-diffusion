from typing import List

import enum
from collections import defaultdict

import numpy as np
import torch
from lark import Token
from lark.parsers.lalr_interactive_parser import InteractiveParser

from td.grammar import Grammar
from td.environments import Environment
from td.learning.gpt import TreeDiffusion
from td.learning.tokenizer import Tokenizer
from td.samplers.mutator import AddParents, Mutation


class DecoderState(object):
    class States(enum.Enum):
        POSITION = 0
        TOKEN = 1
        END = 2

    def __init__(self, grammar: Grammar, tokenizer: Tokenizer, context: str):
        self._grammar = grammar
        self._context = context
        self._tokenizer = tokenizer

        self._tokenized_context, self._real_positions_to_token_positions = (
            self._tokenizer._tokenize_one(context, translate_positions=True)
        )

        self._tree = grammar.parse(context) if context else None
        if self._tree is not None:
            AddParents().visit(self._tree)

        self._interactive_parser: InteractiveParser = None
        nodes = (
            [x for x in self._tree.iter_subtrees()] if self._tree is not None else []
        )
        self._position_to_nodes = defaultdict(list)

        for n in nodes:
            self._position_to_nodes[n.meta.start_pos].append(n)

        self._edit_spans = {x.meta.start_pos: x.meta.end_pos for x in nodes}
        self._possible_starts = set(self._edit_spans.keys())
        self._possible_start_tokens = np.array(
            [
                self._tokenizer.position_token(
                    self._real_positions_to_token_positions[x]
                )
                for x in self._possible_starts
            ]
        )
        self._token_positions_to_real = {
            self._real_positions_to_token_positions[x]: x for x in self._possible_starts
        }
        self._start_tokens_mask = (
            self._mask_from_indexes(self._possible_start_tokens)
            if len(self._possible_start_tokens) > 0
            else None
        )
        self._valid_tokens = None
        self._valid_tokens_mask = None

        self._decode_state = DecoderState.States.POSITION

        self._current_start = None
        self._current_end = None
        self._feeded_tokens = []
        self._position_probs = None

    def _mask_from_indexes(self, indexes: np.ndarray):
        mask = np.zeros(self._tokenizer.vocabulary_size)
        mask[indexes] = 1
        return mask

    @property
    def mask(self):
        if self._decode_state == DecoderState.States.POSITION:
            return self._start_tokens_mask

        return self._valid_tokens_mask

    def _node_rule(self, node):
        if not hasattr(node, "parent"):
            return self._grammar._start_name

        idx = node.parent.children.index(node)
        matched = self._grammar.tree_matcher.match_tree(node.parent, node.parent.data)
        rule_name = matched.children[idx].data
        return rule_name

    def _position_rule(self, position):
        nodes = self._position_to_nodes[position]
        rule_names = [self._node_rule(x) for x in nodes]

        for rule_name in rule_names:
            options = self._grammar._nonterminals[rule_name]
            if len(options) > 1:
                return rule_name

        return rule_names[0]

    def _recompute_mask(self):
        accepts = self._interactive_parser.accepts()

        if len(accepts) == 1 and list(accepts)[0] == "$END":
            self._decode_state = DecoderState.States.END
            self._valid_tokens = np.array([self._tokenizer.eos_token])
            self._valid_tokens_mask = self._mask_from_indexes(self._valid_tokens)
            return

        valid_tokens = [
            self._tokenizer._token_to_index[self._grammar.vocabulary_map[x]]
            for x in accepts
        ]
        self._valid_tokens = np.array(valid_tokens)
        self._valid_tokens_mask = self._mask_from_indexes(self._valid_tokens)

    def _edit_probs(self):
        rv = []
        for start_token in self._possible_start_tokens:
            prob = self._position_probs[start_token]
            start = self._token_positions_to_real[
                self._tokenizer.token_to_position(start_token)
            ]
            end = self._edit_spans[start]
            rv.append((start, end, prob))
        return rv

    def feed_token(self, token: int, probs=None):
        # Does not check if token is valid, assumes it is.
        if self._decode_state == DecoderState.States.POSITION:
            self._position_probs = probs
            self._current_start = self._token_positions_to_real[
                self._tokenizer.token_to_position(token)
            ]
            self._current_end = self._edit_spans[self._current_start]
            self._decode_state = DecoderState.States.TOKEN
            position_rule = self._position_rule(self._current_start)
            self._interactive_parser = self._grammar._lark_parser_for_start[
                position_rule
            ].parse_interactive(start=position_rule)
            self._recompute_mask()
        elif self._decode_state == DecoderState.States.TOKEN:
            token_str = self._tokenizer._index_to_token[token]
            token_name = self._grammar.rev_vocabulary_map[token_str]
            self._feeded_tokens.append(token_str)
            feed_token = Token(token_name, token_str)
            self._interactive_parser.feed_token(feed_token)
            self._recompute_mask()

    def force_set_rule(self, rule_name: str = None):
        self._decode_state = DecoderState.States.TOKEN

        if rule_name is None:
            rule_name = self._grammar.start_symbol.name

        self._interactive_parser = self._grammar._lark_parser_for_start[
            rule_name
        ].parse_interactive(start=rule_name)
        self._recompute_mask()

    def get_mutation(self):
        return Mutation(
            self._current_start,
            self._current_end,
            "".join(self._feeded_tokens),
            edit_probs=None if self._position_probs is None else self._edit_probs(),
        )

    def get_feeded_string(self):
        return "".join(self._feeded_tokens)


def expressions_to_images(
    env: Environment,
    expressions: List[str],
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.tensor(np.array([env.compile(x) for x in expressions]))
        .float()
        .permute(0, 3, 1, 2)
        .to(device)
    )


def sample_model_kv(
    model: TreeDiffusion,
    env: Environment,
    tokenizer: Tokenizer,
    current_expressions,
    target_images,
    current_images=None,
    return_probs=False,
    temperature=1.0,
    blank_current=False,
) -> List[Mutation]:
    with torch.inference_mode():
        device = next(model.parameters()).device

        current_images = (
            torch.tensor(np.array([env.compile(x) for x in current_expressions]))
            .float()
            .permute(0, 3, 1, 2)
            .to(device)
            if current_images is None
            else current_images
        )
        if blank_current:
            current_images = current_images * 0.0

        image_embeddings = model.image_embeddings(target_images, current_images)
        context_tokens = [tokenizer._tokenize_one(x) for x in current_expressions]
        start_decoding_positions = [len(x) for x in context_tokens]
        # Pad to max length.
        max_length = tokenizer.max_token_length
        current_tokens = torch.tensor(
            [
                x
                + [tokenizer.sos_token]
                + [tokenizer.pad_token] * (max_length - len(x) - 1)
                for x in context_tokens
            ]
        ).to(device)
        start_decoding_positions = torch.tensor(start_decoding_positions)
        decode_states = [
            DecoderState(env.grammar, tokenizer, x) for x in current_expressions
        ]
        current_position = 0
        k_cache = None
        v_cache = None

        while (
            not all(x._decode_state == DecoderState.States.END for x in decode_states)
            and current_position < max_length - 1
        ):
            logits, k_cache, v_cache = model.transformer(
                current_tokens[:, [current_position]],
                extra_emb=image_embeddings,
                k_cache=k_cache,
                v_cache=v_cache,
                start_idx=current_position,
            )
            logits = logits.cpu()

            logits_for_positions = logits[:, 0, :]
            logits_for_positions = logits_for_positions / temperature
            decode_mask = torch.tensor(np.stack([x.mask for x in decode_states])).bool()
            logits_for_positions = torch.where(
                decode_mask, logits_for_positions, -torch.tensor(float("inf"))
            )
            probs = torch.nn.functional.softmax(logits_for_positions, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)

            # Update decode states.
            for i, token in enumerate(np.array(sampled_tokens)):
                if current_position < start_decoding_positions[i]:
                    continue

                if decode_states[i]._decode_state == DecoderState.States.END:
                    continue

                decode_states[i].feed_token(
                    token,
                    probs=probs[i].numpy()
                    if return_probs
                    and decode_states[i]._decode_state == DecoderState.States.POSITION
                    else None,
                )
                current_tokens[i, current_position + 1] = sampled_tokens[i].item()

            # Update current tokens and positions.
            current_position += 1

        return [x.get_mutation() if not return_probs else x for x in decode_states]


def ar_decoder(
    model: TreeDiffusion,
    env: Environment,
    tokenizer: Tokenizer,
    num_image_tokens: int,
    target_images,
    temperature=1.0,
) -> List[Mutation]:
    with torch.inference_mode():
        device = next(model.parameters()).device
        image_embeddings = model.image_embeddings(target_images, target_images * 0.0)
        start_decoding_positions = [num_image_tokens] * len(target_images)
        # Pad to max length.
        max_length = tokenizer.max_token_length
        current_tokens = torch.tensor(
            [
                [tokenizer.pad_token] * num_image_tokens
                + [tokenizer.sos_token]
                + [tokenizer.pad_token] * (max_length - num_image_tokens - 1)
                for _ in target_images
            ]
        ).to(device)
        start_decoding_positions = torch.tensor(start_decoding_positions)
        decode_states = [
            DecoderState(env.grammar, tokenizer, "") for _ in target_images
        ]

        for decode_state in decode_states:
            decode_state.force_set_rule()

        current_position = 0
        k_cache = None
        v_cache = None

        while (
            not all(x._decode_state == DecoderState.States.END for x in decode_states)
            and current_position < max_length - 1
        ):
            logits, k_cache, v_cache = model.transformer(
                current_tokens[:, [current_position]],
                extra_emb=image_embeddings,
                k_cache=k_cache,
                v_cache=v_cache,
                start_idx=current_position,
            )
            logits = logits.cpu()

            logits_for_positions = logits[:, 0, :]
            logits_for_positions = logits_for_positions / temperature
            decode_mask = torch.tensor(np.stack([x.mask for x in decode_states])).bool()
            logits_for_positions = torch.where(
                decode_mask, logits_for_positions, -torch.tensor(float("inf"))
            )
            probs = torch.nn.functional.softmax(logits_for_positions, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)

            # Update decode states.
            for i, token in enumerate(np.array(sampled_tokens)):
                if current_position < start_decoding_positions[i]:
                    continue

                if decode_states[i]._decode_state == DecoderState.States.END:
                    continue

                decode_states[i].feed_token(token)
                current_tokens[i, current_position + 1] = sampled_tokens[i].item()

            # Update current tokens and positions.
            current_position += 1

        return [x.get_feeded_string() for x in decode_states]
