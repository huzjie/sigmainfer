# -*- coding: utf-8 -*-
"""多轮会话演示。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.engine import Session, SessionManager
from sigmainfer.tokenizer import ByteTokenizer

tok = ByteTokenizer()
engine = SigmaEngine(SigmaModel(), GenerationConfig(max_new_tokens=16, block_size=8))
s = Session(engine)
r1 = s.chat(tok.encode("第一轮"))
r2 = s.chat(tok.encode("第二轮"))
print("历史轮次:", len(s.history))
print("第二轮输出 token 数:", len(r2))
