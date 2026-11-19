# -*- coding: utf-8 -*-
import pytest

from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig


@pytest.fixture
def model():
    return SigmaModel()


@pytest.fixture
def engine():
    return SigmaEngine(SigmaModel(), GenerationConfig(
        max_new_tokens=32, block_size=8, num_steps=4))
