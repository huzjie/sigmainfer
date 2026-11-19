# -*- coding: utf-8 -*-
"""生成事件类型（供回调 / 流式 / 观测使用）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class UnmaskEvent:
    block: int
    step: int
    positions: List[int]
    num: int = field(init=False)

    def __post_init__(self):
        self.num = len(self.positions)


@dataclass
class BlockDoneEvent:
    block: int
    steps_used: int
    tokens: List[int]
