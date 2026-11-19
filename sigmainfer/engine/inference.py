# -*- coding: utf-8 -*-
"""SigmaEngine：分块扩散推理入口。"""
from __future__ import annotations

import random
from typing import List, Optional

from ..core import (GenerationState, get_schedule, get_sampler,
                    ConfidenceThreshold, Stopwatch, mask_tokens)
from ..model import SigmaModel
from .config import GenerationConfig
from .scheduler import run_block, plan_blocks


class SigmaEngine:
    def __init__(self, model: SigmaModel, cfg: GenerationConfig = None):
        self.model = model
        self.cfg = cfg or GenerationConfig()

    def generate(self, prompt_ids: List[int],
                 cfg: GenerationConfig = None) -> GenerationState:
        cfg = cfg or self.cfg
        rng = random.Random(cfg.seed)
        sampler = get_sampler(cfg.sampler, temperature=cfg.temperature, k=cfg.top_k)
        schedule = get_schedule(cfg.schedule, cfg.num_steps)
        threshold = ConfidenceThreshold(
            initial=cfg.confidence_threshold,
            min_value=cfg.confidence_min,
            decay=cfg.confidence_decay,
        )
        state = GenerationState(prompt_ids=list(prompt_ids))
        prefix = list(prompt_ids)
        watch = Stopwatch()

        for size in plan_blocks(cfg):
            block = __import__("sigmainfer.core", fromlist=["Block"]).Block(
                start=len(prefix), ids=mask_tokens(size))
            run_block(prefix, block, self.model, cfg, sampler, schedule,
                      threshold, rng, state)
            generated = block.ids
            state.generated.extend(generated)
            state.block_step_counts.append(block.step)
            state.total_blocks += 1
            state.block_index += 1
            prefix.extend(generated)

        state.wall_seconds = watch.elapsed()
        return state
