# -*- coding: utf-8 -*-
"""FastAPI 服务（可选依赖）。"""
from __future__ import annotations

import json
from typing import List, Optional

from .engine import SigmaEngine, GenerationConfig
from .tokenizer import get_tokenizer


def build_app(engine: SigmaEngine):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("serve 需要 fastapi/uvicorn：pip install -e '.[serve]'") from e

    app = FastAPI(title="SigmaInfer", version="0.1.0")

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: Optional[int] = None
        block_size: Optional[int] = None
        num_steps: Optional[int] = None
        sampler: Optional[str] = None
        seed: Optional[int] = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/generate")
    def generate(req: GenerateRequest):
        tok = get_tokenizer("byte")
        cfg = engine.cfg
        override = {}
        for key in ("max_new_tokens", "block_size", "num_steps", "sampler", "seed"):
            val = getattr(req, key)
            if val is not None:
                override[key] = val
        cfg = GenerationConfig(**{**cfg.as_dict(), **override})
        state = engine.generate(tok.encode(req.prompt), cfg)
        return {"output": tok.decode(state.generated), "stats": state.snapshot()}

    return app


def run_server(engine: SigmaEngine, host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("serve 需要 uvicorn：pip install -e '.[serve]'") from e
    app = build_app(engine)
    uvicorn.run(app, host=host, port=port)
