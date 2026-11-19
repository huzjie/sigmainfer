# 配置参考

## 模型（model）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| vocab_size | 256 | 词表大小 |
| dim | 64 | 隐藏维度 |
| n_heads | 4 | 注意力头数 |
| n_layers | 2 | Transformer 层数 |
| ffn_mult | 4 | FFN 隐藏维度 = dim * ffn_mult |
| max_seq | 1024 | 最大序列长度 |
| seed | 42 | 随机权重种子 |

## 生成（generation）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| max_new_tokens | 64 | 最多生成 token 数 |
| block_size | 16 | 每块 token 数 |
| num_steps | 8 | 每块去噪步数 |
| schedule | cosine | cosine/linear/constant |
| sampler | confidence | confidence/topk/argmax |
| temperature | 1.0 | 采样温度 |
| confidence_threshold | 0.7 | 门槛 |
| confidence_min | 0.2 | 门槛下限 |
| confidence_decay | 0.9 | 退火衰减 |
| top_k | 4 | TopK 每步 unmask 数 |
| seed | 42 | 采样种子 |

## 示例 JSON

```json
{
  "model": {"vocab_size": 256, "dim": 64, "n_heads": 4, "n_layers": 2},
  "generation": {"max_new_tokens": 64, "block_size": 16, "num_steps": 8}
}
```
