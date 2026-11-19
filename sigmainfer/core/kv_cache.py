# -*- coding: utf-8 -*-
"""KV Cache：跨 block 复用前缀的注意力键值缓存。

分块扩散中，已生成的 block 会作为后续 block 的前缀。若每次都重算前缀的
key/value，计算量随长度线性膨胀。KV cache 把已算好的 key/value 存起来，
新 block 只算新增位置的注意力。本实现为纯 Python 参考版本。
"""
from __future__ import annotations

from typing import Dict, List, Optional


class KVCache:
    """按 (layer, position) 存储 key/value 向量。"""

    def __init__(self, enable: bool = True, max_len: int = 4096):
        self.enable = enable
        self.max_len = max_len
        self._keys: Dict[int, Dict[int, List[float]]] = {}
        self._values: Dict[int, Dict[int, List[float]]] = {}
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._length = 0

    def has(self, layer: int, pos: int) -> bool:
        return self.enable and layer in self._keys and pos in self._keys[layer]

    def get(self, layer: int, pos: int) -> Optional[tuple]:
        if not self.has(layer, pos):
            return None
        return self._keys[layer][pos], self._values[layer][pos]

    def put(self, layer: int, pos: int, key: List[float], value: List[float]) -> None:
        if not self.enable:
            return
        if self._length >= self.max_len:
            return
        self._keys.setdefault(layer, {})[pos] = key
        self._values.setdefault(layer, {})[pos] = value
        self._length = max(self._length, pos + 1)

    def stats(self) -> dict:
        return {
            "enabled": self.enable,
            "length": self._length,
            "max_len": self.max_len,
        }
