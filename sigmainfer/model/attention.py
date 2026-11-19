# -*- coding: utf-8 -*-
"""多头注意力（双向，适合扩散式并行预测）。"""
from __future__ import annotations

import math
import random
from typing import List, Sequence

from .config import ModelConfig
from ._ops import matvec, softmax, zeros, scale


class MultiHeadAttention:
    def __init__(self, cfg: ModelConfig, rng: random.Random):
        d = cfg.dim
        scale_w = (1.0 / d) ** 0.5
        g = lambda: [[rng.gauss(0, scale_w) for _ in range(d)] for _ in range(d)]  # noqa: E731
        self.Wq, self.Wk, self.Wv = g(), g(), g()
        self.Wo = [[rng.gauss(0, scale_w) for _ in range(d)] for _ in range(d)]
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

    def _split_heads(self, v: Sequence[float]) -> List[List[float]]:
        return [v[h * self.head_dim:(h + 1) * self.head_dim]
                for h in range(self.n_heads)]

    def _merge_heads(self, hs: List[List[float]]) -> List[float]:
        out = []
        for h in hs:
            out.extend(h)
        return out

    def __call__(self, x: Sequence[Sequence[float]]) -> List[List[float]]:
        n = len(x)
        qs = [matvec(self.Wq, xi) for xi in x]
        ks = [matvec(self.Wk, xi) for xi in x]
        vs = [matvec(self.Wv, xi) for xi in x]
        scale_d = 1.0 / math.sqrt(self.head_dim)
        outs = []
        for qi in qs:
            head_out = []
            for h in range(self.n_heads):
                qh = qi[h * self.head_dim:(h + 1) * self.head_dim]
                scores = []
                for kj in ks:
                    kh = kj[h * self.head_dim:(h + 1) * self.head_dim]
                    scores.append(sum(a * b for a, b in zip(qh, kh)) * scale_d)
                probs = softmax(scores)
                acc = zeros(self.head_dim)
                for j, p in enumerate(probs):
                    vh = vs[j][h * self.head_dim:(h + 1) * self.head_dim]
                    for k in range(self.head_dim):
                        acc[k] += p * vh[k]
                head_out.extend(acc)
            outs.append(matvec(self.Wo, head_out))
        return outs
