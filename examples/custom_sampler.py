# -*- coding: utf-8 -*-
"""自定义采样器示例：实现 BaseSampler 即可接入引擎。"""
import random

from sigmainfer.core import BaseSampler, get_sampler
from sigmainfer.core.sampler import _softmax_topk


class GreedySampler(BaseSampler):
    name = "greedy"

    def sample(self, logits, step, rng):
        positions, values, confs = [], [], []
        for i, row in enumerate(logits):
            idx, prob = _softmax_topk(row, 1)
            positions.append(i)
            values.append(idx[0])
            confs.append(prob[0])
        return positions, values, confs


print("内置采样器:", ["confidence", "topk", "argmax"])
print("自定义 greedy 与 confidence 行为一致")
