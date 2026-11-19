# -*- coding: utf-8 -*-
"""KV Cache 基本用法演示。"""
from sigmainfer.core import KVCache

cache = KVCache(enable=True)
cache.put(0, 0, [1.0, 2.0], [3.0, 4.0])
print("has(0,0):", cache.has(0, 0))
print("get(0,0):", cache.get(0, 0))
print("stats:", cache.stats())
