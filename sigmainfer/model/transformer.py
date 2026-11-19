# -*- coding: utf-8 -*-
"""Transformer 块：注意力 + 前馈，各带残差与 LayerNorm。"""
from __future__ import annotations

import random
from typing import Sequence

from .config import ModelConfig
from .attention import MultiHeadAttention
from .ffn import FFN
from ._ops import add, layer_norm


class TransformerBlock:
    def __init__(self, cfg: ModelConfig, rng: random.Random):
        d = cfg.dim
        self.attn = MultiHeadAttention(cfg, rng)
        self.ffn = FFN(cfg, rng)
        self.ln1_g = [1.0] * d
        self.ln1_b = [0.0] * d
        self.ln2_g = [1.0] * d
        self.ln2_b = [0.0] * d

    def __call__(self, x):
        # 注意力 + 残差
        h = self.attn(x)
        x = [add(xi, hi) for xi, hi in zip(x, h)]
        # 前馈 + 残差
        normed = [layer_norm(xi, self.ln1_g, self.ln1_b) for xi in x]
        f = self.ffn(normed)
        x = [add(xi, fi) for xi, fi in zip(x, f)]
        return x
