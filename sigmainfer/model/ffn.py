# -*- coding: utf-8 -*-
"""前馈网络（FFN）：dim -> ffn_dim -> dim，GELU 近似激活。"""
from __future__ import annotations

import math
import random
from typing import List, Sequence

from .config import ModelConfig
from ._ops import matvec


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class FFN:
    def __init__(self, cfg: ModelConfig, rng: random.Random):
        d, f = cfg.dim, cfg.ffn_dim
        s1 = (1.0 / d) ** 0.5
        s2 = (1.0 / f) ** 0.5
        self.W1 = [[rng.gauss(0, s1) for _ in range(d)] for _ in range(f)]
        self.b1 = [0.0] * f
        self.W2 = [[rng.gauss(0, s2) for _ in range(f)] for _ in range(d)]
        self.b2 = [0.0] * d

    def __call__(self, x: Sequence[Sequence[float]]) -> List[List[float]]:
        outs = []
        for xi in x:
            h = [gelu(v) for v in matvec(self.W1, xi)]
            outs.append(matvec(self.W2, h))
        return outs
