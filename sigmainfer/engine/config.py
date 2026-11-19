# -*- coding: utf-8 -*-
"""生成配置（GenerationConfig）。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64       # 最多生成多少 token
    block_size: int = 16           # 每个扩散块的大小
    num_steps: int = 8             # 每个块的最大去噪步数
    schedule: str = "cosine"       # cosine / linear / constant
    sampler: str = "confidence"    # confidence / topk / argmax
    temperature: float = 1.0
    confidence_threshold: float = 0.7
    confidence_min: float = 0.2
    confidence_decay: float = 0.9
    top_k: int = 4                 # TopKSampler 每步 unmask 数
    seed: int = 42
    use_kv_cache: bool = False     # 参考实现：关闭，仅保留开关

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})
