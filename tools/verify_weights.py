# -*- coding: utf-8 -*-
"""工具：校验权重文件 SHA-256。"""
import sys

from sigmainfer.utils.checksum import sha256_hex

if len(sys.argv) < 2:
    print("用法: python verify_weights.py <weights-file>")
    sys.exit(1)
print("sha256:", sha256_hex(sys.argv[1]))
