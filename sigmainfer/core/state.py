# -*- coding: utf-8 -*-
"""GenerationState：一次完整生成过程的运行时状态。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .tokens import TokenId


@dataclass
class GenerationState:
    """记录一次 generate() 调用从头到尾的可观测状态。

    包含已生成前缀、当前块、步数统计与事件流，方便 benchmark / 报告 / 复现。
    """

    prompt_ids: List[TokenId] = field(default_factory=list)
    generated: List[TokenId] = field(default_factory=list)
    block_index: int = 0
    total_steps: int = 0
    total_blocks: int = 0
    # 事件流：记录每步 unmask 了哪些位置，供回溯与可视化
    events: List[dict] = field(default_factory=list)
    # 每个块实际消耗的去噪步数（用于观测「早停」效果）
    block_step_counts: List[int] = field(default_factory=list)

    def record_unmask(self, block: int, step: int, positions: List[int]) -> None:
        self.events.append({
            "block": block,
            "step": step,
            "positions": positions,
            "num": len(positions),
        })

    def snapshot(self) -> dict:
        return {
            "prompt_len": len(self.prompt_ids),
            "generated_len": len(self.generated),
            "total_blocks": self.total_blocks,
            "total_steps": self.total_steps,
            "block_step_counts": self.block_step_counts,
        }
