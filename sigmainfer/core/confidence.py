# -*- coding: utf-8 -*-
"""置信度门槛 + 自适应退火。

分块扩散的「早停」关键：每步只 unmask 置信度 >= threshold 的位置。
如果某步 unmask 太少（甚至为 0），说明门槛太严，逐步调低 threshold；
如果 unmask 得很快，可适当调高以省步数。这就是「置信度退火」。
"""
from __future__ import annotations

from typing import List, Sequence


class ConfidenceThreshold:
    def __init__(self, initial: float = 0.7, min_value: float = 0.2,
                 decay: float = 0.9, stall_steps: int = 2):
        self.initial = initial
        self.min_value = min_value
        self.decay = decay
        self.stall_steps = stall_steps
        self.current = initial
        self._stall = 0

    def filter(self, positions: List[int], values: List[int],
               confidences: Sequence[float]) -> tuple:
        """按当前门槛过滤：仅保留置信度达标的决策。"""
        keep_p, keep_v, keep_c = [], [], []
        for p, v, c in zip(positions, values, confidences):
            if c >= self.current:
                keep_p.append(p)
                keep_v.append(v)
                keep_c.append(c)
        return keep_p, keep_v, keep_c

    def update(self, num_unmasked: int, num_masked_before: int) -> None:
        """根据本轮 unmask 数量自适应调整门槛。"""
        if num_unmasked == 0:
            self._stall += 1
        else:
            self._stall = 0
        if self._stall >= self.stall_steps:
            self.current = max(self.min_value, self.current * self.decay)
            self._stall = 0

    def reset(self) -> None:
        self.current = self.initial
        self._stall = 0
