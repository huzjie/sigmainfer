# 接入指南

## 接入你的模型

实现 `sigmainfer/model/sigma_model.py` 的权重加载：从 safetensors 读入并映射到
`Embedding / TransformerBlock / DiffusionHead` 张量结构，`forward` 保持
`ids -> logits` 契约不变。

## 接入你的服务

`SigmaEngine.generate(prompt_ids, cfg)` 是唯一入口，返回 `GenerationState`。
Web/SDK/CLI 全部复用该入口，替换引擎即可。

## 接入你的评测

`benchmark/compare.py` 展示如何在同一引擎下切换 block_size 做对比，
照此模式可扩展你自己的指标。
