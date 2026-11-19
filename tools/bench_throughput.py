# -*- coding: utf-8 -*-
"""工具：扫不同 block_size 的吞吐。"""
import sys

from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.benchmark.compare import _measure
from sigmainfer.tokenizer import ByteTokenizer

tok = ByteTokenizer()
prompt = tok.encode("throughput sweep")
for bs in (8, 16, 32, 64):
    cfg = GenerationConfig(max_new_tokens=64, block_size=bs, num_steps=8)
    m = _measure(SigmaEngine(SigmaModel(), cfg), prompt, cfg)
    print(f"block_size={bs:3d}  tps={m.tokens_per_second:8.2f}  "
          f"steps_per_block={m.steps_per_block:5.3f}")
sys.exit(0)
