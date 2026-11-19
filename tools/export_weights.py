# -*- coding: utf-8 -*-
"""工具：导出随机权重清单（演示）。"""
import sys

from sigmainfer.model import SigmaModel
from sigmainfer.model.weights import export_weights

model = SigmaModel()
export_weights(model, "weights.json")
print("已导出 weights.json（仅含 config + seed，参考实现）")
sys.exit(0)
