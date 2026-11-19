# -*- coding: utf-8 -*-
"""Sigma 文本扩散模型：接收含掩码的序列，输出每个位置的词表 logits。"""
from __future__ import annotations

from typing import List, Sequence

from .config import ModelConfig
from .embedding import Embedding
from .transformer import TransformerBlock
from .head import DiffusionHead
from .weights import init_weights


class SigmaModel:
    """简化扩散 Transformer（随机权重参考实现）。

    与自回归模型不同：输入序列中未生成的位置用 MASK_ID 占位，模型对整条序列
    做双向编码，输出每个位置的词表分布；推理侧只取掩码位置做「并行 unmask」。
    """

    def __init__(self, cfg: ModelConfig = None):
        self.cfg = cfg or ModelConfig()
        self.cfg.validate()
        self.embedding, self.layers, self.head = init_weights(self.cfg)

    def forward(self, ids: Sequence[int]) -> List[List[float]]:
        """返回 shape = (len(ids), vocab_size) 的 logits 列表。"""
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def logits_for_masked(self, ids: Sequence[int], masked_positions: Sequence[int]):
        """只取掩码位置的对数概率，供采样器使用。"""
        all_logits = self.forward(ids)
        return [all_logits[p] for p in masked_positions]
