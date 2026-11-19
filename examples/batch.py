# -*- coding: utf-8 -*-
"""批量生成示例。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.engine import generate_batch
from sigmainfer.tokenizer import ByteTokenizer

engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=32, block_size=16))
tok = ByteTokenizer()
prompts = [tok.encode(p) for p in ["a", "b", "c"]]
results, agg = generate_batch(engine, prompts)
print("batch:", len(results), "tokens:", agg.generated_tokens,
      "tps:", round(agg.tokens_per_second, 2))
