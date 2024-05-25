from td.grammar import Grammar
import datrie


class Tokenizer(object):
    def __init__(
        self,
        grammar: Grammar,
        max_token_length: int = 256,
        max_sequence_length: int = 256,
        include_partial: bool = False,
        include_positions: bool = True,
    ):
        self._grammar = grammar
        self._vocabulary = ["<PAD>", "<SOS>", "<EOS>"] + grammar.vocabulary

        if include_positions:
            self._vocabulary += [f"<POS {i}>" for i in range(max_token_length)]

        if include_partial:
            self._partial_tokens = [
                f"<{str(x)}>" for x in self._grammar.nonterminals.keys()
            ]
            self._vocabulary += self._partial_tokens
            self._partial_tokens_set = set(self._partial_tokens)

        self._include_partial = include_partial
        self._include_positions = include_positions

        self._vocabulary_size = len(self._vocabulary)
        self._vocabulary_set = set(self._vocabulary)
        self._max_token_length = max(len(token) for token in self._vocabulary)
        self._characters = sorted(list(set("".join(self._vocabulary))))
        self._max_token_length = max_token_length
        self._max_sequence_length = max_sequence_length

        self._token_to_index = {token: i for i, token in enumerate(self._vocabulary)}
        self._index_to_token = {i: token for i, token in enumerate(self._vocabulary)}

        self._trie = datrie.Trie(self._characters)
        for token, index in self._token_to_index.items():
            self._trie[token] = index

        self._pad_token = self._token_to_index["<PAD>"]
        self._sos_token = self._token_to_index["<SOS>"]
        self._eos_token = self._token_to_index["<EOS>"]

    def _prefix_match_trie(self, current_expression: str):
        return self._trie.longest_prefix(current_expression)

    def _prefix_match(self, current_expression: str):
        for i in range(min(len(current_expression), self._max_token_length), 0, -1):
            prefix = current_expression[:i]
            if prefix in self._vocabulary_set:
                return prefix

        return None

    def _tokenize_one(
        self, expression: str, translate_positions: bool = False, pad: bool = False
    ):
        token_indexes = []
        current_expression = expression

        if translate_positions:
            positions = []

        while current_expression:
            match = self._prefix_match_trie(current_expression)
            if match:
                token_indexes.append(self._token_to_index[match])
                current_expression = current_expression[len(match) :]
                if translate_positions:
                    positions.extend([len(token_indexes) - 1] * len(match))
            else:
                raise ValueError(f"Unrecognized token: {current_expression}")

        if pad:
            token_indexes += [self._pad_token] * (
                self._max_sequence_length - len(token_indexes)
            )

        if translate_positions:
            return token_indexes, positions

        return token_indexes

    def _untokenize_one(self, token_indexes):
        return "".join(self._index_to_token[i] for i in token_indexes)

    def position_token(self, position: int):
        assert 0 <= position < self._max_token_length, f"Invalid position: {position}"
        return self._token_to_index[f"<POS {position}>"]

    def token_to_position(self, token_id: int):
        token = self._index_to_token[token_id]
        return int(token[4:-1])

    @property
    def grammar(self):
        return self._grammar

    @property
    def vocabulary(self):
        return self._grammar.vocabulary

    @property
    def max_token_length(self):
        return self._max_token_length

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def sos_token(self):
        return self._sos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def partial_tokens(self):
        return self._partial_tokens

    @property
    def partial_tokens_set(self):
        return self._partial_tokens_set
