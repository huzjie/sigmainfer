# -*- coding: utf-8 -*-
"""Token primitives and constants."""

from typing import List, Union

TokenId = int
# 掩码位置用 -1 表示「尚未生成」；模型的嵌入层额外分配一行给 [MASK]。
MASK_ID = -1

# 特殊语义 token（字节级 tokenizer 之外的控制符）
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def is_mask(tok: TokenId) -> bool:
    """判断一个 token 是否为掩码（待生成）状态。"""
    return tok == MASK_ID


def mask_tokens(length: int) -> List[TokenId]:
    """返回指定长度的全掩码序列。"""
    return [MASK_ID] * length
