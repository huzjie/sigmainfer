# -*- coding: utf-8 -*-
"""工具：对比三种 schedule 的步数分布。"""
import sys

from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

tok = ByteTokenizer()
prompt = tok.encode("schedule comparison")
for sched in ("cosine", "linear", "constant"):
    cfg = GenerationConfig(max_new_tokens=64, block_size=16, num_steps=8,
                           schedule=sched, seed=42)
    st = SigmaEngine(SigmaModel(), cfg).generate(prompt)
    print(f"{sched:8s}  steps={st.total_steps:3d}  block_steps={st.block_step_counts}")
sys.exit(0)
