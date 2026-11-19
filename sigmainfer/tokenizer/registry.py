# -*- coding: utf-8 -*-
"""Tokenizer 注册表。"""
from __future__ import annotations

from .base import Tokenizer
from .byte_tokenizer import ByteTokenizer
from .char_tokenizer import CharTokenizer

_REGISTRY = {
    "byte": ByteTokenizer,
    "char": CharTokenizer,
}


def get_tokenizer(kind: str = "byte", **kwargs) -> Tokenizer:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown tokenizer: {kind}")
    return _REGISTRY[kind](**kwargs)


def list_tokenizers() -> list:
    return sorted(_REGISTRY)
