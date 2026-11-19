# -*- coding: utf-8 -*-
"""工具：输出一次生成的完整指标 JSON。"""
import json
import sys

from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.tokenizer import ByteTokenizer

engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=64, block_size=16))
st = engine.generate(ByteTokenizer().encode("metrics dump"))
print(json.dumps(st.snapshot(), ensure_ascii=False, indent=2))
sys.exit(0)
