# 架构设计

## 分层

```
cli / serve / sdk            ← 入口层
        │
engine (inference/streaming/batch/session/scheduler)   ← 推理编排层
        │
core (block/schedule/sampler/confidence/kv_cache)      ← 分块扩散原语层
        │
model (embedding/attention/ffn/head/transformer)       ← 扩散 Transformer 层
        │
tokenizer (byte/char)        ← 文本 <-> token 层
```

## 数据流

1. `tokenizer.encode(prompt)` -> `prompt_ids`。
2. `engine.scheduler.plan_blocks` 把 `max_new_tokens` 切成若干 block 大小。
3. 每个 block：`model.forward(full_sequence)` -> 全位置 logits，取掩码位置喂给采样器。
4. 采样器输出候选 + 置信度，`ConfidenceThreshold` 过滤后 unmask。
5. 掩码调度控制每步 unmask 比例，早停 + 兜底收尾。
6. `GenerationState` 记录所有可观测事件，供 benchmark / 报告消费。

## 设计原则

- **零依赖内核**：core/model/engine 只用标准库，任意环境可跑。
- **采样/调度可替换**：sampler、schedule 都走注册表，接口单一。
- **随机权重参考实现**：缩小规模（dim=64/2 层/词表 256）保证任意机器可跑通，
  换真实权重只需替换 model 层。
- **早停 + 兜底**：门槛退火让快的块早停，argmax 兜底让慢的块一定出结果。
