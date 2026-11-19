# -*- coding: utf-8 -*-
"""生成回调：在 unmask / block 完成时触发。"""
from __future__ import annotations

from typing import Callable, List

UnmaskCallback = Callable[[int, int, List[int]], None]
BlockCallback = Callable[[int, int, List[int]], None]


class CallbackRegistry:
    def __init__(self):
        self.on_unmask: List[UnmaskCallback] = []
        self.on_block_done: List[BlockCallback] = []

    def unmask(self, fn: UnmaskCallback) -> UnmaskCallback:
        self.on_unmask.append(fn)
        return fn

    def block_done(self, fn: BlockCallback) -> BlockCallback:
        self.on_block_done.append(fn)
        return fn

    def fire_unmask(self, block: int, step: int, positions: List[int]) -> None:
        for fn in self.on_unmask:
            fn(block, step, positions)

    def fire_block_done(self, block: int, steps: int, tokens: List[int]) -> None:
        for fn in self.on_block_done:
            fn(block, steps, tokens)
