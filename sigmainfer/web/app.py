# -*- coding: utf-8 -*-
"""Web 控制台：挂载静态页 + 复用 serve 的 API。"""
from __future__ import annotations

import os

from ..engine import SigmaEngine
from ..serve import build_app


def build_web_app(engine: SigmaEngine):
    app = build_app(engine)
    try:
        from fastapi.staticfiles import StaticFiles
    except ImportError:  # pragma: no cover
        return app
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
