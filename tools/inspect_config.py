# -*- coding: utf-8 -*-
"""工具：打印模型/生成配置。"""
import sys

from sigmainfer.model import ModelConfig
from sigmainfer.engine import GenerationConfig

mc = ModelConfig()
gc = GenerationConfig()
print("model:", mc.as_dict())
print("generation:", gc.as_dict())
sys.exit(0)
