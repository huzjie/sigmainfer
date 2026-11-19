# -*- coding: utf-8 -*-
"""轻量剖析：统计各阶段耗时（无外部依赖）。"""
from __future__ import annotations

import time
from collections import defaultdict


class StageProfiler:
    def __init__(self):
        self._starts = {}
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self._starts:
            return
        dt = time.perf_counter() - self._starts.pop(name)
        self._totals[name] += dt
        self._counts[name] += 1

    def report(self) -> dict:
        return {k: {"total": round(v, 4), "calls": self._counts[k]}
                for k, v in sorted(self._totals.items(), key=lambda x: -x[1])}
