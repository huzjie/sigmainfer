# -*- coding: utf-8 -*-
"""块循环调度：把「一次 generate」拆成可观测的 block 序列。

分块扩散的调度本质是一个双层循环：
  外层 for block in blocks      —— 决定生成到哪、块多大
  内层 for step in steps        —— 决定这个块里每步 unmask 多少
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..core import (Block, GenerationState, get_schedule, get_sampler,
                    ConfidenceThreshold, MASK_ID)


def run_block(prefix, block: Block, model, cfg, sampler, schedule,
              threshold: ConfidenceThreshold, rng: random.Random,
              state: GenerationState) -> None:
    """对一个 block 执行完整的去噪循环（原地修改 block.ids）。"""
    for step in range(cfg.num_steps):
        masked = block.masked_positions
        if not masked:
            break
        full = prefix + block.ids
        full_positions = [block.start + p for p in masked]
        logits = model.logits_for_masked(full, full_positions)
        rel_positions, values, confs = sampler.sample(logits, step, rng)
        keep_p, keep_v, keep_c = threshold.filter(rel_positions, values, confs)
        if keep_p:
            local = [masked[i] for i in keep_p]
            block.unmask(local, keep_v)
            state.record_unmask(state.block_index, step, local)
        block.step = step + 1
        state.total_steps += 1
        threshold.update(len(keep_v), len(masked))
        if block.all_unmasked:
            break

    # 兜底：步数耗尽仍有掩码 -> 强制 argmax 收尾，保证「一定出结果」
    if not block.all_unmasked:
        masked = block.masked_positions
        full = prefix + block.ids
        logits = model.logits_for_masked(full, [block.start + p for p in masked])
        for i, p in enumerate(masked):
            best = max(range(len(logits[i])), key=lambda j: logits[i][j])
            block.ids[p] = best
        state.record_unmask(state.block_index, cfg.num_steps, masked)
    block.freeze()


def plan_blocks(cfg) -> List[int]:
    """按 max_new_tokens 与 block_size 规划各块大小。"""
    n = cfg.max_new_tokens
    sizes = []
    while n > 0:
        s = min(cfg.block_size, n)
        sizes.append(s)
        n -= s
    return sizes
