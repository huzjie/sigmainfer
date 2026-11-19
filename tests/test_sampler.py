# -*- coding: utf-8 -*-
import random

from sigmainfer.core.sampler import ConfidenceSampler, TopKSampler, ArgMaxSampler


def test_confidence_returns_all():
    logits = [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]
    s = ConfidenceSampler()
    p, v, c = s.sample(logits, 0, random.Random(0))
    assert p == [0, 1]
    assert v == [2, 0]


def test_topk_limits():
    logits = [[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]]
    s = TopKSampler(k=2)
    p, v, c = s.sample(logits, 0, random.Random(0))
    assert len(p) == 2


def test_argmax_deterministic():
    logits = [[0.5, 0.5], [1.0, 0.0]]
    s = ArgMaxSampler()
    p, v, _ = s.sample(logits, 0, random.Random(0))
    assert v[1] == 0
