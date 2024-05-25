import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    output_size: int = None  # Use vocab_size if None
    causal: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        # feature projections
        self.kqv_projection = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.is_causal = config.causal

    def forward(self, x, k_cache=None, v_cache=None):
        B, T, C = x.size()

        # calculate q, k v
        q, k, v = self.kqv_projection(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if k_cache is not None:
            assert v_cache is not None
            assert T == 1
            # concat previous cache with new k, v
            v = torch.cat([v_cache, v], dim=2)  # (B, nh, 1 + T', hs)
            k = torch.cat([k_cache, k], dim=2)  # (B, nh, 1 + T', hs)
            att = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.size(-1))
            )  # (B, nh, 1, 1 + T')
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        return self.proj(y), k, v


class Block(nn.Module):
    """Transfromer Block"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.mlp_sequence = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x, k_cache=None, v_cache=None):
        _x = x
        x, k, v = self.attn(self.ln(x), k_cache, v_cache)
        x = _x + x
        x = x + self.mlp_sequence(x)
        return x, k, v


class Transformer(nn.Module):
    """Simple Transformer"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.block_size = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.n_embd)
        self.transformer_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.layer_norm = nn.LayerNorm(config.n_embd)

        output_size = (
            config.vocab_size if config.output_size is None else config.output_size
        )

        self.lm_head = nn.Linear(config.n_embd, output_size, bias=False)

    def forward(self, idx, extra_emb=None, start_idx=0, k_cache=None, v_cache=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        pos = (
            torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) + start_idx
        )

        tok_emb = self.token_embeddings(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.position_embeddings(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        if extra_emb is not None:
            x = x + extra_emb

        k_s = []
        v_s = []
        for idx, block in enumerate(self.transformer_blocks):
            if k_cache is not None:
                assert v_cache is not None
                k_i, v_i = k_cache[idx], v_cache[idx]
            else:
                k_i, v_i = None, None
            x, k, v = block(x, k_i, v_i)
            k_s.append(k)
            v_s.append(v)

        logits = self.lm_head(self.layer_norm(x))
        return logits, k_s, v_s


class TreeDiffusion(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        input_channels: int = 3,
        image_model_name: str = "resnet18",
    ):
        super().__init__()

        self.transformer = Transformer(config)
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=False,
            in_chans=input_channels * 3,
            num_classes=config.n_embd,
        )
        # n_image_features = self.image_encoder.num_features
        # self.target_proj = nn.Linear(n_image_features, config.n_embd)
        # self.mutated_proj = nn.Linear(n_image_features, config.n_embd)

    def image_embeddings(self, target_images, mutated_images):
        # target_images: (B, C, H, W)
        # mutated_images: (B, C, H, W)

        # Make a bigger batch by having all images in the same batch.
        all_images = torch.cat(
            [target_images, mutated_images, torch.abs(target_images - mutated_images)],
            dim=1,
        )
        all_images = (all_images * 2) - 1  # Normalize to [-1, 1]
        all_features = self.image_encoder(all_images)

        return all_features[:, None]

    def forward(self, idx, target_images, mutated_images):
        image_emb = self.image_embeddings(target_images, mutated_images)
        logits, _, _ = self.transformer(idx, image_emb)
        return logits


class ValueNet(nn.Module):
    def __init__(self, image_model_name, input_channels=3):
        super().__init__()
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=False,
            in_chans=input_channels * 3,
            num_classes=1,
        )

    def forward(self, target_images, mutated_images):
        all_images = torch.cat(
            [target_images, mutated_images, torch.abs(target_images - mutated_images)],
            dim=1,
        )
        all_images = (all_images * 2) - 1
        return self.image_encoder(all_images)


class ValueHead(nn.Module):
    def __init__(
        self,
        n_embd: int = 128,
        n_layers=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(n_embd, n_embd) for _ in range(n_layers)]
        )
        self.value_head = nn.Linear(n_embd, 1)

    def forward(self, image_embeddings):
        x = image_embeddings
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.value_head(x)
