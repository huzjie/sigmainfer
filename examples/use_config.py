# -*- coding: utf-8 -*-
"""从 JSON 配置加载并生成。"""
from sigmainfer.config_loader import load_config
from sigmainfer import SigmaModel, SigmaEngine
from sigmainfer.tokenizer import ByteTokenizer

mc, gc = load_config("config/default.json")
engine = SigmaEngine(SigmaModel(mc), gc)
st = engine.generate(ByteTokenizer().encode("config demo"))
print(ByteTokenizer().decode(st.generated))
