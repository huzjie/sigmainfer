# -*- coding: utf-8 -*-
"""核采样（top-p）过滤：只保留累计概率 <= p 的候选。"""
from __future__ import annotations

import math
from typing import List, Sequence


class NucleusFilter:
    def __init__(self, top_p: float = 0.9):
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.top_p = top_p

    def filter_logits(self, logits: Sequence[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        out = [float("-inf")] * len(logits)
        acc = 0.0
        for i in order:
            out[i] = logits[i]
            acc += probs[i]
            if acc >= self.top_p:
                break
        return out
