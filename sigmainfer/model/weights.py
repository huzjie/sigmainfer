# -*- coding: utf-8 -*-
"""随机权重初始化 + 序列化（JSON）。"""
from __future__ import annotations

import json
import random

from .config import ModelConfig
from .embedding import Embedding
from .transformer import TransformerBlock
from .head import DiffusionHead


def init_weights(cfg: ModelConfig):
    rng = random.Random(cfg.seed)
    emb = Embedding(cfg, rng)
    layers = [TransformerBlock(cfg, rng) for _ in range(cfg.n_layers)]
    head = DiffusionHead(cfg, rng)
    return emb, layers, head


def export_weights(model, path: str) -> None:
    """把参考实现权重导出为 JSON（仅演示/基准用，非真实权重格式）。"""
    payload = {
        "config": model.cfg.as_dict(),
        "seed": model.cfg.seed,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_weights(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
