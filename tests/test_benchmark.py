# -*- coding: utf-8 -*-
from sigmainfer import SigmaModel, SigmaEngine
from sigmainfer.benchmark import run_comparison


def test_comparison_keys():
    engine = SigmaEngine(SigmaModel())
    report = run_comparison(engine, [1, 2, 3], max_new_tokens=16, block_size=8)
    assert "diffusion" in report
    assert "autoregressive" in report
    assert "speedup_tokens_per_sec" in report
