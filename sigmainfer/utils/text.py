# -*- coding: utf-8 -*-
"""文本工具：token 计数 / 截断。"""
from __future__ import annotations


def count_tokens(text: str) -> int:
    """按 UTF-8 字节数估算 token 数（字节级 tokenizer 口径）。"""
    return len(text.encode("utf-8"))


def truncate(text: str, max_bytes: int) -> str:
    bs = text.encode("utf-8")
    if len(bs) <= max_bytes:
        return text
    return bs[:max_bytes].decode("utf-8", errors="ignore")
