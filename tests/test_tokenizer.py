# -*- coding: utf-8 -*-
from sigmainfer.tokenizer import ByteTokenizer, CharTokenizer


def test_byte_roundtrip():
    t = ByteTokenizer()
    ids = t.encode("hello, 世界")
    assert t.decode(ids) == "hello, 世界"


def test_byte_vocab():
    t = ByteTokenizer()
    assert t.vocab_size == 256


def test_char_roundtrip():
    t = CharTokenizer()
    ids = t.encode("abc")
    assert t.decode(ids) == "abc"
