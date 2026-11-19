# -*- coding: utf-8 -*-
"""激活函数：GELU / ReLU / SwiGLU。"""
from __future__ import annotations

import math


def relu(x: float) -> float:
    return max(0.0, x)


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


ACTIVATIONS = {"relu": relu, "gelu": gelu, "silu": silu}


def get_activation(name: str):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation: {name}")
    return ACTIVATIONS[name]
