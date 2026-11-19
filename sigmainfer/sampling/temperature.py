# -*- coding: utf-8 -*-
"""温度缩放：把 logits 除以温度。"""
from __future__ import annotations

from typing import List, Sequence


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def scale(self, logits: Sequence[float]) -> List[float]:
        if self.temperature == 1.0:
            return list(logits)
        return [v / self.temperature for v in logits]
