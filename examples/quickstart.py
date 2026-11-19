# -*- coding: utf-8 -*-
"""最小可运行示例：构建模型 + 生成。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

model = SigmaModel()                       # 随机权重参考实现
engine = SigmaEngine(model, GenerationConfig(max_new_tokens=32, block_size=16))
tok = ByteTokenizer()

prompt = "Once upon a time"
state = engine.generate(tok.encode(prompt))
print("prompt:", prompt)
print("output:", tok.decode(state.generated))
print("stats:", state.snapshot())
