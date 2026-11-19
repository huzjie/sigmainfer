# -*- coding: utf-8 -*-
"""Block（生成块）模型：分块扩散的基本调度单元。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .tokens import TokenId, is_mask


@dataclass
class Block:
    """一段待生成/已生成的 token 区间。

    Attributes:
        start: 在完整序列中的绝对起始位置（含 prompt 前缀长度）。
        ids: 当前 token 序列；未生成位置为 MASK_ID。
        step: 当前已执行的去噪步数。
        frozen: 该块是否已完全 unmask（进入下一块或结束）。
    """

    start: int
    ids: List[TokenId]
    step: int = 0
    frozen: bool = False

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def masked_positions(self) -> List[int]:
        return [i for i, t in enumerate(self.ids) if is_mask(t)]

    @property
    def num_masked(self) -> int:
        return sum(1 for t in self.ids if is_mask(t))

    @property
    def all_unmasked(self) -> bool:
        return self.num_masked == 0

    def unmask(self, positions: List[int], values: List[TokenId]) -> None:
        for pos, val in zip(positions, values):
            self.ids[pos] = val

    def freeze(self) -> None:
        self.frozen = True
