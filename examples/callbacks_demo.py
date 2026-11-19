# -*- coding: utf-8 -*-
"""生成回调演示：观测每块步数。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.engine.callbacks import CallbackRegistry
from sigmainfer.tokenizer import ByteTokenizer

reg = CallbackRegistry()
reg.block_done(lambda block, steps, tokens: print(f"block {block} done in {steps} steps"))

engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=32, block_size=8))
st = engine.generate(ByteTokenizer().encode("callbacks"))
for cb in reg.on_block_done:
    cb(0, st.block_step_counts[0], st.generated[:8])
