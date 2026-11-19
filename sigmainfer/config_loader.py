# -*- coding: utf-8 -*-
"""配置加载：JSON 配置 -> ModelConfig + GenerationConfig。

优先 JSON（无歧义、可机器校验）；也支持 `key: value` 平铺文件。
"""
from __future__ import annotations

import json
from typing import Tuple

from .model import ModelConfig
from .engine import GenerationConfig


def _parse_flat(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        out[k] = _coerce(v)
    return out


def _coerce(v: str):
    v = v.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip('"').strip("'")


def load_config(path: str) -> Tuple[ModelConfig, GenerationConfig]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        data = json.loads(text)
    else:
        data = _parse_flat(text)
    model_cfg = ModelConfig.from_dict(data.get("model", {}))
    gen_cfg = GenerationConfig.from_dict(data.get("generation", {}))
    return model_cfg, gen_cfg


def dump_config(model_cfg: ModelConfig, gen_cfg: GenerationConfig) -> str:
    return json.dumps({
        "model": model_cfg.as_dict(),
        "generation": gen_cfg.as_dict(),
    }, ensure_ascii=False, indent=2)
