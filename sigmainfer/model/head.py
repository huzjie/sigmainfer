# -*- coding: utf-8 -*-
"""扩散头（LM head）：隐藏向量 -> 词表 logits。"""
from __future__ import annotations

import random
from typing import Sequence

from .config import ModelConfig
from ._ops import matvec


class DiffusionHead:
    def __init__(self, cfg: ModelConfig, rng: random.Random):
        d = cfg.dim
        s = (1.0 / d) ** 0.5
        self.W = [[rng.gauss(0, s) for _ in range(d)] for _ in range(cfg.vocab_size)]
        self.b = [0.0] * cfg.vocab_size

    def __call__(self, x: Sequence[Sequence[float]]):
        return [matvec(self.W, xi) for xi in x]
