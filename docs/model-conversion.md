# 模型权重转换

把真实 Sigma safetensors 映射到本参考实现的张量结构：

1. 用 `safetensors` 读取权重张量。
2. 映射到 `Embedding.token`（vocab+1 × dim）、`TransformerBlock.attn.*`、
   `TransformerBlock.ffn.*`、`DiffusionHead.W`（dim × vocab）。
3. 保持 `SigmaModel.forward(ids) -> logits` 契约不变。
4. 加载前用 `sha256_hex` 校验文件完整性。
