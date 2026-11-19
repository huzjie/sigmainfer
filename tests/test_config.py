# -*- coding: utf-8 -*-
import json
import tempfile

from sigmainfer.config_loader import load_config, dump_config
from sigmainfer.model import ModelConfig
from sigmainfer.engine import GenerationConfig


def test_dump_load():
    mc = ModelConfig()
    gc = GenerationConfig()
    text = dump_config(mc, gc)
    data = json.loads(text)
    assert data["model"]["dim"] == 64


def test_load_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"model": {"dim": 32}, "generation": {"block_size": 8}}')
    mc, gc = load_config(str(p))
    assert mc.dim == 32
    assert gc.block_size == 8
