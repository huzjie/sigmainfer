# -*- coding: utf-8 -*-
"""吞吐 / 延迟 / 步数统计，供 benchmark 与报告使用。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    prompt_len: int = 0
    generated_tokens: int = 0
    total_blocks: int = 0
    total_steps: int = 0
    wall_seconds: float = 0.0
    block_step_counts: list = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.wall_seconds

    @property
    def steps_per_block(self) -> float:
        if self.total_blocks <= 0:
            return 0.0
        return self.total_steps / self.total_blocks

    def as_dict(self) -> dict:
        return {
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "total_blocks": self.total_blocks,
            "total_steps": self.total_steps,
            "wall_seconds": round(self.wall_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "steps_per_block": round(self.steps_per_block, 3),
            "block_step_counts": self.block_step_counts,
        }


class Stopwatch:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
