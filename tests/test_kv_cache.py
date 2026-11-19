# -*- coding: utf-8 -*-
from sigmainfer.core import KVCache


def test_put_get():
    c = KVCache()
    c.put(0, 0, [1.0], [2.0])
    assert c.has(0, 0)
    assert c.get(0, 0) == ([1.0], [2.0])


def test_disabled():
    c = KVCache(enable=False)
    c.put(0, 0, [1.0], [2.0])
    assert not c.has(0, 0)
