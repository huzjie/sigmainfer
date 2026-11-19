# -*- coding: utf-8 -*-
from sigmainfer import SigmaModel
from sigmainfer.model import ModelConfig


def test_forward_shape():
    m = SigmaModel(ModelConfig(vocab_size=256, dim=32, n_heads=4, n_layers=1))
    logits = m.forward([1, 2, 3, -1])
    assert len(logits) == 4
    assert len(logits[0]) == 256


def test_logits_for_masked():
    m = SigmaModel()
    logits = m.logits_for_masked([1, -1, 2, -1], [1, 3])
    assert len(logits) == 2
