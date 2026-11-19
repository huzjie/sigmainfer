# -*- coding: utf-8 -*-
from sigmainfer.core.confidence import ConfidenceThreshold


def test_filter():
    th = ConfidenceThreshold(initial=0.5)
    p, v, _ = th.filter([0, 1, 2], [10, 20, 30], [0.9, 0.3, 0.8])
    assert p == [0, 2]
    assert v == [10, 30]


def test_anneal_on_stall():
    th = ConfidenceThreshold(initial=0.7, min_value=0.2, decay=0.5, stall_steps=2)
    th.update(0, 10)
    th.update(0, 10)
    assert th.current < 0.7


def test_min_value_floor():
    th = ConfidenceThreshold(initial=0.3, min_value=0.2, decay=0.1, stall_steps=1)
    for _ in range(10):
        th.update(0, 10)
    assert th.current >= th.min_value
