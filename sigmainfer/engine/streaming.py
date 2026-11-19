# -*- coding: utf-8 -*-
"""流式生成：以「块」为粒度逐块产出 token。

分块扩散天然以 block 为产出单位（块内并行、块间串行），因此流式接口按块
yield，而不是逐 token。对下游「打字机」式展示，展开每块即可。
"""
from __future__ import annotations

from typing import Generator, List

from .inference import SigmaEngine
from .config import GenerationConfig


def stream_generate(engine: SigmaEngine, prompt_ids: List[int],
                    cfg: GenerationConfig = None) -> Generator[List[int], None, None]:
    cfg = cfg or engine.cfg
    state = engine.generate(prompt_ids, cfg)
    # 按 block_step_counts 切分 generated，逐块 yield
    cursor = 0
    for size in plan_blocks_of(cfg):
        yield state.generated[cursor:cursor + size]
        cursor += size


def plan_blocks_of(cfg: GenerationConfig) -> List[int]:
    n = cfg.max_new_tokens
    sizes = []
    while n > 0:
        s = min(cfg.block_size, n)
        sizes.append(s)
        n -= s
    return sizes
