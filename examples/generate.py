# -*- coding: utf-8 -*-
"""带自定义配置的生成示例。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

cfg = GenerationConfig(
    max_new_tokens=64,
    block_size=16,
    num_steps=8,
    schedule="cosine",
    sampler="confidence",
    confidence_threshold=0.7,
    seed=7,
)
engine = SigmaEngine(SigmaModel(), cfg)
tok = ByteTokenizer()
state = engine.generate(tok.encode("扩散生成演示"))
print(tok.decode(state.generated))
print("blocks:", state.total_blocks, "steps:", state.total_steps,
      "step_counts:", state.block_step_counts)
