# -*- coding: utf-8 -*-
"""会话管理：带历史的多轮生成（供 serve / SDK 使用）。"""
from __future__ import annotations

import uuid
from typing import List

from .inference import SigmaEngine
from .config import GenerationConfig


class Session:
    def __init__(self, engine: SigmaEngine, session_id: str = None):
        self.id = session_id or uuid.uuid4().hex[:12]
        self.engine = engine
        self.history: List[dict] = []

    def chat(self, prompt_ids: List[int], cfg: GenerationConfig = None) -> List[int]:
        # 拼接历史作为前缀，简化多轮上下文
        prefix: List[int] = []
        for turn in self.history:
            prefix.extend(turn["prompt"])
            prefix.extend(turn["reply"])
        prefix.extend(prompt_ids)
        state = self.engine.generate(prefix, cfg)
        self.history.append({"prompt": prompt_ids, "reply": state.generated})
        return state.generated


class SessionManager:
    def __init__(self, engine: SigmaEngine):
        self.engine = engine
        self._sessions = {}

    def create(self) -> Session:
        s = Session(self.engine)
        self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError(f"unknown session: {session_id}")
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_ids(self) -> List[str]:
        return list(self._sessions)
