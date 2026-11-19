# -*- coding: utf-8 -*-
"""RoPE 旋转位置编码（参考实现，默认关闭，可接入注意力）。"""
from __future__ import annotations

import math
from typing import List, Tuple


def precompute(dim: int, max_seq: int, base: float = 10000.0) -> Tuple[List, List]:
    freqs = [1.0 / (base ** (2 * (i // 2) / dim)) for i in range(dim)]
    cos, sin = [], []
    for pos in range(max_seq):
        cos.append([math.cos(pos * f) for f in freqs])
        sin.append([math.sin(pos * f) for f in freqs])
    return cos, sin


def rotate(x: List[float], pos: int, cos, sin) -> List[float]:
    out = []
    for i in range(0, len(x), 2):
        c, s = cos[pos][i], sin[pos][i]
        x0, x1 = x[i], x[i + 1]
        out.append(x0 * c - x1 * s)
        out.append(x0 * s + x1 * c)
    return out
