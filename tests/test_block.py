# -*- coding: utf-8 -*-
from sigmainfer.core import Block, MASK_ID


def test_block_mask():
    b = Block(start=0, ids=[MASK_ID, MASK_ID, MASK_ID])
    assert b.num_masked == 3
    assert b.masked_positions == [0, 1, 2]


def test_unmask():
    b = Block(start=0, ids=[MASK_ID, MASK_ID])
    b.unmask([0, 1], [5, 6])
    assert b.all_unmasked
    assert b.ids == [5, 6]
