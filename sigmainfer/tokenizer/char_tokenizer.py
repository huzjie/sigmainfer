# -*- coding: utf-8 -*-
"""字符级 tokenizer：按 Unicode 码点建词表。"""
from __future__ import annotations

from typing import List, Iterable

from .base import Tokenizer


class CharTokenizer(Tokenizer):
    name = "char"

    def __init__(self, corpus: str = None):
        if corpus:
            chars = sorted(set(corpus))
        else:
            chars = [chr(c) for c in range(32, 127)]  # 可打印 ASCII
        self._c2i = {ch: i for i, ch in enumerate(chars)}
        self._i2c = {i: ch for ch, i in self._c2i.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._c2i)

    def encode(self, text: str) -> List[int]:
        return [self._c2i.get(ch, self._c2i.get("?", 0)) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._i2c.get(i, "?") for i in ids)

    def build_vocab(self, texts: Iterable[str]) -> None:
        chars = sorted(set("".join(texts)))
        self._c2i = {ch: i for i, ch in enumerate(chars)}
        self._i2c = {i: ch for ch, i in self._c2i.items()}
