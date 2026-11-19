# -*- coding: utf-8 -*-
"""报告渲染：把生成状态渲染成 JSON / Markdown / 文本。"""
from __future__ import annotations

import json

from .core import GenerationState, RunMetrics


def to_json(state: GenerationState, metrics: RunMetrics = None) -> str:
    payload = {"state": state.snapshot()}
    if metrics is not None:
        payload["metrics"] = metrics.as_dict()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_markdown(state: GenerationState, metrics: RunMetrics = None) -> str:
    lines = ["# 生成报告", "", f"- prompt 长度：{len(state.prompt_ids)}",
             f"- 生成 token：{len(state.generated)}",
             f"- 总块数：{state.total_blocks}",
             f"- 总步数：{state.total_steps}",
             f"- 各块步数：{state.block_step_counts}"]
    if metrics is not None:
        m = metrics.as_dict()
        lines += ["", "## 性能", f"- 吞吐：{m['tokens_per_second']} token/s",
                  f"- 平均每块步数：{m['steps_per_block']}"]
    return "\n".join(lines)


def to_text(state: GenerationState, metrics: RunMetrics = None) -> str:
    return to_markdown(state, metrics).replace("#", "").replace("*", "")
