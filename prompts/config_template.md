# 配置模板（JSON）

```json
{
  "model": {
    "vocab_size": 256,
    "dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_mult": 4,
    "max_seq": 1024,
    "seed": 42
  },
  "generation": {
    "max_new_tokens": 64,
    "block_size": 16,
    "num_steps": 8,
    "schedule": "cosine",
    "sampler": "confidence",
    "temperature": 1.0,
    "confidence_threshold": 0.7,
    "confidence_min": 0.2,
    "confidence_decay": 0.9,
    "top_k": 4,
    "seed": 42
  }
}
```
