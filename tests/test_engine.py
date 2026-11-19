# -*- coding: utf-8 -*-
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig


def test_generate_length():
    cfg = GenerationConfig(max_new_tokens=16, block_size=8, num_steps=4)
    engine = SigmaEngine(SigmaModel(), cfg)
    state = engine.generate([1, 2, 3])
    assert len(state.generated) == 16
    assert state.total_blocks == 2


def test_generate_all_unmasked():
    cfg = GenerationConfig(max_new_tokens=8, block_size=8, num_steps=4)
    engine = SigmaEngine(SigmaModel(), cfg)
    state = engine.generate([1, 2, 3])
    assert all(t != -1 for t in state.generated)


def test_block_step_counts():
    cfg = GenerationConfig(max_new_tokens=16, block_size=8, num_steps=4)
    engine = SigmaEngine(SigmaModel(), cfg)
    state = engine.generate([1, 2, 3])
    assert len(state.block_step_counts) == 2
    assert all(1 <= c <= 4 for c in state.block_step_counts)
