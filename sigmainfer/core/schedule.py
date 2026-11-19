# -*- coding: utf-8 -*-
"""掩码调度（mask schedule）：决定每一步保留多少比例的掩码。

分块扩散的核心之一：一个 block 从「全掩码」出发，经过 T 步去噪逐步 unmask。
每步「还剩多少位置处于掩码」由 schedule 决定，常见三种：
- cosine：早期大步 unmask、后期精细收敛（MaskGIT 风格）。
- linear：每步均匀 unmask 1/T。
- constant：每步按固定比例 unmask。
"""
from __future__ import annotations

import math


class MaskSchedule:
    def __init__(self, kind: str = "cosine", total_steps: int = 8,
                 keep_ratio: float = 0.5):
        if kind not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown schedule kind: {kind}")
        self.kind = kind
        self.total_steps = max(1, total_steps)
        self.keep_ratio = keep_ratio

    def keep_fraction(self, step: int) -> float:
        """返回第 step 步（0-indexed）之后仍应保持掩码的比例（0~1）。"""
        t = step / max(1, self.total_steps - 1)
        if self.kind == "cosine":
            # cos(pi/2 * t)：t=0 -> 1.0，t=1 -> 0.0
            return math.cos(math.pi / 2.0 * t)
        if self.kind == "linear":
            return 1.0 - t
        # constant：每步固定 unmask keep_ratio
        return max(0.0, 1.0 - self.keep_ratio * step)

    def unmask_fraction(self, step: int) -> float:
        return 1.0 - self.keep_fraction(step)


SCHEDULES = {
    "cosine": lambda steps, kr: MaskSchedule("cosine", steps, kr),
    "linear": lambda steps, kr: MaskSchedule("linear", steps, kr),
    "constant": lambda steps, kr: MaskSchedule("constant", steps, kr),
}


def get_schedule(kind: str, total_steps: int, keep_ratio: float = 0.5) -> MaskSchedule:
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule kind: {kind}")
    return SCHEDULES[kind](total_steps, keep_ratio)
