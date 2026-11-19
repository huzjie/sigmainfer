# -*- coding: utf-8 -*-
"""模型工厂：按配置构建 / 从权重清单加载。"""
from __future__ import annotations

from .config import ModelConfig
from .sigma_model import SigmaModel
from .weights import load_weights
from ..core.registry import MODEL_REGISTRY, register_model


@register_model("sigma-reference")
def build_sigma_reference(cfg: dict = None) -> SigmaModel:
    return SigmaModel(ModelConfig.from_dict(cfg or {}))


def build_model(name: str = "sigma-reference", cfg: dict = None) -> SigmaModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {name}")
    return MODEL_REGISTRY[name](cfg)


def build_from_checkpoint(path: str) -> SigmaModel:
    meta = load_weights(path)
    return SigmaModel(ModelConfig.from_dict(meta.get("config", {})))
