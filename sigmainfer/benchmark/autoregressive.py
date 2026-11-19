# -*- coding: utf-8 -*-
"""自回归基线：block_size=1、num_steps=1，逐 token 串行产出。"""
from __future__ import annotations

from ..engine import SigmaEngine, GenerationConfig


def autoregressive_engine(model) -> SigmaEngine:
    cfg = GenerationConfig(block_size=1, num_steps=1, sampler="argmax")
    return SigmaEngine(model, cfg)
