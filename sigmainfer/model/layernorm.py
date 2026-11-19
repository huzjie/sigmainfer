# -*- coding: utf-8 -*-
"""LayerNorm 独立实现（供扩展使用）。"""
from __future__ import annotations

import math
from typing import List, Sequence


def layer_norm(x: Sequence[float], gamma: Sequence[float],
               beta: Sequence[float], eps: float = 1e-5) -> List[float]:
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv * g + b for xi, g, b in zip(x, gamma, beta)]


def rms_norm(x: Sequence[float], gamma: Sequence[float], eps: float = 1e-5) -> List[float]:
    n = len(x)
    ms = sum(xi * xi for xi in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [xi * inv * g for xi, g in zip(x, gamma)]
