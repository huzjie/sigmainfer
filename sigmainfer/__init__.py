# -*- coding: utf-8 -*-
"""SigmaInfer — Sigma 文本扩散推理引擎（分块扩散，并行生成 token）。"""
from .model import ModelConfig, SigmaModel, build_model
from .engine import GenerationConfig, SigmaEngine
from .core import MASK_ID

__version__ = "0.1.0"

__all__ = [
    "__version__", "ModelConfig", "SigmaModel", "build_model",
    "GenerationConfig", "SigmaEngine", "MASK_ID",
]
