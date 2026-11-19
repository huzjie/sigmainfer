# 采样与置信度指南

## 三种采样器

- `confidence`：每位置 argmax，置信度 = softmax 概率，交给门槛统一过滤。
- `topk`：每步只对置信度最高的 k 个位置做决策，其余保持掩码。
- `argmax`：确定性，无温度，用于可复现基准。

## 置信度门槛退火

```
initial=0.7  每步只 unmask 置信度 >= 0.7 的位置
min_value=0.2  门槛最低降到 0.2
decay=0.9  停滞时 threshold *= 0.9
stall_steps=2  连续 2 步 unmask 为 0 才退火
```

## 调参顺序建议

1. 先固定 `block_size`、`num_steps` 跑吞吐。
2. 再调 `confidence_threshold` 保质量（看早停步数）。
3. 最后用 `topk` 加速，用 `argmax` 复现基准。
