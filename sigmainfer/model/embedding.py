# -*- coding: utf-8 -*-
"""Token + 位置嵌入。掩码位置映射到嵌入矩阵的额外一行。"""
from __future__ import annotations

import random
from typing import List, Sequence

from .config import ModelConfig
from ._ops import add


class Embedding:
    def __init__(self, cfg: ModelConfig, rng: random.Random):
        self.cfg = cfg
        d = cfg.dim
        scale = (1.0 / d) ** 0.5
        # 词表 + 1（MASK 占一行）
        self.token = [[rng.gauss(0, scale) for _ in range(d)]
                      for _ in range(cfg.vocab_size + 1)]
        self.position = [[rng.gauss(0, scale) for _ in range(d)]
                         for _ in range(cfg.max_seq)]

    def _map(self, tok: int) -> int:
        if tok == -1:  # MASK_ID
            return self.cfg.vocab_size
        return tok

    def __call__(self, ids: Sequence[int]) -> List[List[float]]:
        return [add(self.token[self._map(t)], self.position[i])
                for i, t in enumerate(ids)]
