# -*- coding: utf-8 -*-
import math

from sigmainfer.core.schedule import get_schedule


def test_cosine_bounds():
    s = get_schedule("cosine", 8)
    assert abs(s.keep_fraction(0) - 1.0) < 1e-6
    assert abs(s.keep_fraction(7)) < 1e-6


def test_linear():
    s = get_schedule("linear", 4)
    assert abs(s.keep_fraction(0) - 1.0) < 1e-6
    assert abs(s.keep_fraction(3)) < 1e-6


def test_unknown_schedule():
    import pytest
    with pytest.raises(ValueError):
        get_schedule("nope", 8)
