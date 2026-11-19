# -*- coding: utf-8 -*-
"""前缀预计算（prefill）：对 prompt 前缀做一次前向，缓存 KV。

参考实现中 KV cache 默认关闭；此模块预留 prefill 入口，接真实权重时可启用。
"""
from __future__ import annotations

from typing import List


def prefill_prefix(model, prefix_ids: List[int], kv_cache=None) -> None:
    """对前缀做一次前向，把中间 KV 写入 cache（占位，接真实权重时实现）。"""
    if kv_cache is None:
        return
    model.forward(prefix_ids)
