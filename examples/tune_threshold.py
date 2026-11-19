# -*- coding: utf-8 -*-
"""门槛退火效果演示：不同 initial 的步数分布。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

tok = ByteTokenizer()
prompt = tok.encode("threshold demo")
for initial in (0.5, 0.7, 0.9):
    cfg = GenerationConfig(max_new_tokens=64, block_size=16, num_steps=8,
                           confidence_threshold=initial, seed=42)
    st = SigmaEngine(SigmaModel(), cfg).generate(prompt)
    print(f"initial={initial}  steps={st.total_steps}  "
          f"block_steps={st.block_step_counts}")
