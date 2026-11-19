# -*- coding: utf-8 -*-
"""对比：分块扩散 vs 自回归（吞吐 / 步数 / 延迟）。"""
from __future__ import annotations

from typing import List

from ..engine import SigmaEngine, GenerationConfig
from ..core import RunMetrics, Stopwatch


def _measure(engine: SigmaEngine, prompt_ids: List[int], cfg: GenerationConfig) -> RunMetrics:
    watch = Stopwatch()
    state = engine.generate(prompt_ids, cfg)
    m = RunMetrics(
        prompt_len=len(prompt_ids),
        generated_tokens=len(state.generated),
        total_blocks=state.total_blocks,
        total_steps=state.total_steps,
        wall_seconds=watch.elapsed(),
        block_step_counts=state.block_step_counts,
    )
    return m


def run_comparison(engine: SigmaEngine, prompt_ids: List[int],
                   max_new_tokens: int = 64, block_size: int = 16) -> dict:
    # 扩散
    diff_cfg = GenerationConfig(max_new_tokens=max_new_tokens,
                                block_size=block_size, num_steps=8)
    diff = _measure(engine, prompt_ids, diff_cfg)
    # 自回归基线（block_size=1, num_steps=1）
    ar_cfg = GenerationConfig(max_new_tokens=max_new_tokens,
                              block_size=1, num_steps=1, sampler="argmax")
    ar = _measure(engine, prompt_ids, ar_cfg)

    return {
        "config": {"max_new_tokens": max_new_tokens, "block_size": block_size},
        "diffusion": diff.as_dict(),
        "autoregressive": ar.as_dict(),
        "speedup_tokens_per_sec": round(
            diff.tokens_per_second / ar.tokens_per_second, 3
        ) if ar.tokens_per_second > 0 else None,
    }
