# -*- coding: utf-8 -*-
"""采样器：把模型对掩码位置的 logits 转成「unmask 哪些位置 + 填什么 token」。

所有采样器共享同一接口：输入 logits（n_masked × vocab）与当前步，返回
(positions, token_ids, confidence) 三元组。positions 为 block 内索引，
token_ids 为对应 token，confidence 为每个决策的置信度（用于门槛过滤）。
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def _softmax_topk(logits: Sequence[float], k: int = 1, temperature: float = 1.0):
    """纯 Python softmax + top-k，返回 (indices, probs) 按概率降序。"""
    t = max(temperature, 1e-6)
    scaled = [v / t for v in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps)
    probs = [e / s for e in exps]
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return order[:k], [probs[i] for i in order[:k]]


class BaseSampler:
    name = "base"

    def sample(self, logits: Sequence[Sequence[float]], step: int,
               rng: random.Random) -> Tuple[List[int], List[int], List[float]]:
        raise NotImplementedError


class ConfidenceSampler(BaseSampler):
    """每个掩码位置取 argmax 作为候选，置信度 = softmax 概率。"""

    name = "confidence"

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def sample(self, logits, step, rng):
        positions, values, confs = [], [], []
        for i, row in enumerate(logits):
            idx, prob = _softmax_topk(row, 1, self.temperature)
            positions.append(i)
            values.append(idx[0])
            confs.append(prob[0])
        return positions, values, confs


class TopKSampler(BaseSampler):
    """每步只对置信度最高的 top-k 个掩码位置做出决策（其余保持掩码）。"""

    name = "topk"

    def __init__(self, k: int = 4, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def sample(self, logits, step, rng):
        cands = []
        for i, row in enumerate(logits):
            idx, prob = _softmax_topk(row, 1, self.temperature)
            cands.append((i, idx[0], prob[0]))
        cands.sort(key=lambda c: c[2], reverse=True)
        chosen = cands[: self.k]
        return [c[0] for c in chosen], [c[1] for c in chosen], [c[2] for c in chosen]


class ArgMaxSampler(BaseSampler):
    """无温度 argmax，确定性输出（用于可复现基准）。"""

    name = "argmax"

    def sample(self, logits, step, rng):
        positions, values, confs = [], [], []
        for i, row in enumerate(logits):
            best = max(range(len(row)), key=lambda j: row[j])
            exps = [math.exp(v - max(row)) for v in row]
            s = sum(exps)
            positions.append(i)
            values.append(best)
            confs.append(exps[best] / s if s > 0 else 1.0)
        return positions, values, confs


SAMPLERS = {
    "confidence": ConfidenceSampler,
    "topk": TopKSampler,
    "argmax": ArgMaxSampler,
}


def get_sampler(kind: str, **kwargs) -> BaseSampler:
    if kind not in SAMPLERS:
        raise ValueError(f"unknown sampler kind: {kind}")
    cls = SAMPLERS[kind]
    import inspect
    sig = inspect.signature(cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)
