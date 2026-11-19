# -*- coding: utf-8 -*-
"""计时器。"""
from __future__ import annotations

import time


class Timer:
    def __init__(self):
        self._start = time.perf_counter()
        self._laps = []

    def lap(self, name: str = "") -> float:
        now = time.perf_counter()
        dt = now - self._start
        self._laps.append((name, dt))
        return dt

    def total(self) -> float:
        return time.perf_counter() - self._start

    def as_dict(self) -> dict:
        return {"laps": [{"name": n, "seconds": round(d, 4)}
                         for n, d in self._laps],
                "total": round(self.total(), 4)}
