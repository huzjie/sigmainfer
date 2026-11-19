# -*- coding: utf-8 -*-
"""sigmainfer.sampling — 采样策略扩展（温度 / 核采样）。"""
from .temperature import TemperatureScaler
from .nucleus import NucleusFilter

__all__ = ["TemperatureScaler", "NucleusFilter"]
