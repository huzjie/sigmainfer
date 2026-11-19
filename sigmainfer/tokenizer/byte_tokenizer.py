# -*- coding: utf-8 -*-
"""字节级 tokenizer：UTF-8 编码，词表 = 256（+MASK 额外一行）。"""
from __future__ import annotations

from typing import List

from .base import Tokenizer


class ByteTokenizer(Tokenizer):
    name = "byte"
    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        bs = bytes(b for b in ids if 0 <= b <= 255)
        return bs.decode("utf-8", errors="replace")
