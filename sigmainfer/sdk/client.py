# -*- coding: utf-8 -*-
"""Python SDK 客户端：通过 HTTP 调用远程 sigmainfer 服务。"""
from __future__ import annotations

import json
import urllib.request


class SigmaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, **kwargs) -> dict:
        payload = {"prompt": prompt, **kwargs}
        return self._post("/v1/generate", payload)

    def health(self) -> dict:
        return self._get("/health")

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
