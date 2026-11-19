# -*- coding: utf-8 -*-
"""按块流式输出示例。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.engine import stream_generate
from sigmainfer.tokenizer import ByteTokenizer

engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=48, block_size=16))
tok = ByteTokenizer()
for block in stream_generate(engine, tok.encode("stream demo")):
    print("[block]", tok.decode(block))
