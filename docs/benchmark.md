# 基准对比：自回归 vs 扩散

## 方法

同一引擎、同一模型权重、同一 prompt 下：
- 扩散：`block_size=16, num_steps=8`
- 自回归：`block_size=1, num_steps=1, sampler=argmax`

输出两者 tokens_per_second、steps_per_block、block_step_counts 与 speedup。

## 运行

```bash
sigmainfer benchmark --max-new-tokens 64 --block-size 16
```

## 指标说明

- `tokens_per_second`：生成吞吐。
- `steps_per_block`：平均每块去噪步数（观察早停效果）。
- `speedup_tokens_per_sec`：扩散相对自回归的吞吐倍数。
