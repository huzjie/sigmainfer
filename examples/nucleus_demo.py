# -*- coding: utf-8 -*-
"""核采样（top-p）过滤演示。"""
from sigmainfer.sampling import NucleusFilter

logits = [0.1, 2.5, 1.0, 0.3, 0.05]
f = NucleusFilter(top_p=0.9)
filtered = f.filter_logits(logits)
print("原始:", logits)
print("过滤:", filtered)
