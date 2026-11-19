# -*- coding: utf-8 -*-
"""文件校验：加载权重前先核对 SHA-256。"""
from __future__ import annotations

import hashlib


def sha256_hex(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
