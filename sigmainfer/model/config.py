# -*- coding: utf-8 -*-
"""模型配置。"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 256          # 字节级词表
    dim: int = 64                  # 隐藏维度（参考实现缩小规模）
    n_heads: int = 4               # 注意力头数
    n_layers: int = 2              # Transformer 层数
    ffn_mult: int = 4              # FFN 隐藏维度 = dim * ffn_mult
    max_seq: int = 1024            # 最大序列长度
    seed: int = 42                 # 随机权重种子

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return self.dim * self.ffn_mult

    def validate(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
