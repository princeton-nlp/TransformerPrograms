import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Define network architecture. Adapted from
# https://colab.research.google.com/drive/19gn2tavBGDqOYHLatjSROhABBD5O_JyZ
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model, init_emb=None):
        super().__init__()
        if init_emb is not None:
            self.W_E = nn.Parameter(init_emb)
        else:
            self.W_E = nn.Parameter(
                torch.randn(d_model, d_vocab) / np.sqrt(d_model)
            )

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model, mask=None):
        super().__init__()
        self.W_U = nn.Parameter(
            torch.randn(d_model, d_vocab) / np.sqrt(d_vocab)
        )
        if mask is not None:
            self.register_buffer(
                "mask", torch.tensor(mask).view(1, 1, len(mask))
            )
        else:
            self.mask = None

    def forward(self, x):
        logits = x @ self.W_U
        if self.mask is not None:
            logits = logits.masked_fill(self.mask, -1e30)
        return logits


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(
            torch.randn(max_ctx, d_model) / np.sqrt(d_model)
        )

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class Transformer(nn.Module):
    def __init__(
        self,
        d_vocab,
        n_layers=2,
        d_model=64,
        d_mlp=64,
        n_heads=4,
        n_ctx=32,
        act_type="ReLU",
        d_vocab_out=None,
        dropout=0.0,
        init_emb=None,
        unembed_mask=None,
        pool_outputs=False,
        **kwargs,
    ):
        super().__init__()
        self.embed = Embed(d_vocab, d_model, init_emb=init_emb)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=d_mlp,
            dropout=dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.unembed = Unembed(
            d_vocab_out or d_vocab, d_model, mask=unembed_mask
        )
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.pool_outputs = pool_outputs

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = x * np.sqrt(x.shape[-1])
        x = self.pos_embed(x)
        x = self.dropout(x)
        m = mask
        mask = torch.cat([mask for _ in range(self.n_heads)], 0)
        mask[:, :, 0] = True
        x = self.encoder(x, mask=~mask)
        if self.pool_outputs:
            x = torch.cat(
                [
                    x.masked_fill(~m[:, 0].unsqueeze(-1), 0).mean(
                        1, keepdims=True
                    ),
                    x[:, 1:],
                ],
                1,
            )
        x = self.unembed(x)
        return x

    @property
    def device(self):
        return self.embed.W_E.device
