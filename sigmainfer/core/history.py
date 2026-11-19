# -*- coding: utf-8 -*-
"""生成历史：记录多次生成结果，支持统计与导出。"""
from __future__ import annotations

from typing import List


class GenerationHistory:
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[dict] = []

    def add(self, snapshot: dict) -> None:
        self.entries.append(snapshot)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def summary(self) -> dict:
        if not self.entries:
            return {"runs": 0}
        tokens = [e["generated_len"] for e in self.entries]
        return {
            "runs": len(self.entries),
            "total_tokens": sum(tokens),
            "avg_tokens": sum(tokens) / len(tokens),
        }

    def clear(self) -> None:
        self.entries.clear()
