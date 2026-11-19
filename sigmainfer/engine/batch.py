# -*- coding: utf-8 -*-
"""批量生成：多个 prompt 顺序/批量推理，聚合指标。"""
from __future__ import annotations

from typing import List

from .inference import SigmaEngine
from .config import GenerationConfig
from ..core import RunMetrics, Stopwatch


def generate_batch(engine: SigmaEngine, prompts: List[List[int]],
                   cfg: GenerationConfig = None) -> List[dict]:
    cfg = cfg or engine.cfg
    results = []
    watch = Stopwatch()
    for prompt in prompts:
        state = engine.generate(prompt, cfg)
        results.append({
            "prompt_ids": prompt,
            "generated_ids": state.generated,
            "state": state.snapshot(),
        })
    total = sum(len(r["generated_ids"]) for r in results)
    agg = RunMetrics(
        prompt_len=sum(len(r["prompt_ids"]) for r in results),
        generated_tokens=total,
        total_blocks=sum(r["state"]["total_blocks"] for r in results),
        total_steps=sum(r["state"]["total_steps"] for r in results),
        wall_seconds=watch.elapsed(),
    )
    return results, agg
