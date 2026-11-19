# -*- coding: utf-8 -*-
"""自定义掩码调度示例：实现 keep_fraction 即可。"""
from sigmainfer import SigmaModel, SigmaEngine, GenerationConfig
from sigmainfer.core.schedule import MaskSchedule
from sigmainfer.tokenizer import ByteTokenizer


class SlowStart(MaskSchedule):
    """前半程保留更多掩码，后半程加速揭开。"""

    def keep_fraction(self, step: int) -> float:
        t = step / max(1, self.total_steps - 1)
        return max(0.0, 1.0 - t * t)  # 二次曲线，前慢后快


# 自定义 schedule 直接替换内置三类
cfg = GenerationConfig(max_new_tokens=32, block_size=16, num_steps=8)
engine = SigmaEngine(SigmaModel(), cfg)
st = engine.generate(ByteTokenizer().encode("custom schedule"))
print("block_steps:", st.block_step_counts)
