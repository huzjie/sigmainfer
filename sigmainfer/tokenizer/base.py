# -*- coding: utf-8 -*-
"""Tokenizer 抽象接口。"""
from __future__ import annotations

from typing import List


class Tokenizer:
    name = "base"
    vocab_size = 0

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
