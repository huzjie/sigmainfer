# -*- coding: utf-8 -*-
"""纯 Python 线性代数小工具（零依赖内核专用）。"""
from __future__ import annotations

import math
from typing import List, Sequence


def zeros(n: int) -> List[float]:
    return [0.0] * n


def matvec(M: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    """M: m x n, v: n -> m"""
    out = []
    for row in M:
        s = 0.0
        for a, b in zip(row, v):
            s += a * b
        out.append(s)
    return out


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """A: m x k, B: k x n -> m x n"""
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[float], s: float) -> List[float]:
    return [x * s for x in a]


def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def layer_norm(x: Sequence[float], gamma: Sequence[float],
               beta: Sequence[float], eps: float = 1e-5) -> List[float]:
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv * g + b for xi, g, b in zip(x, gamma, beta)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
