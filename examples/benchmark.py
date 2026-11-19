# -*- coding: utf-8 -*-
"""自回归 vs 扩散对比示例。"""
from sigmainfer import SigmaModel, SigmaEngine
from sigmainfer.benchmark import run_comparison
from sigmainfer.tokenizer import ByteTokenizer

engine = SigmaEngine(SigmaModel())
tok = ByteTokenizer()
report = run_comparison(engine, tok.encode("hello world"),
                        max_new_tokens=64, block_size=16)
print("diffusion tps:", report["diffusion"]["tokens_per_second"])
print("autoregressive tps:", report["autoregressive"]["tokens_per_second"])
print("speedup:", report["speedup_tokens_per_sec"])
