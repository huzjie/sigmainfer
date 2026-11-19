# -*- coding: utf-8 -*-
"""工具：跑一组 prompt，输出各 block 步数分布。"""
import sys

from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=64, block_size=16))
tok = ByteTokenizer()
for p in ["hello world", "diffusion model", "parallel decoding"]:
    st = engine.generate(tok.encode(p))
    print(p, "->", st.block_step_counts, "total_steps=", st.total_steps)
sys.exit(0)
