# 术语表

- **block diffusion**：分块扩散，把生成序列切块、块内并行去噪。
- **mask schedule**：掩码调度，决定每步揭开多少掩码。
- **unmask**：把 [MASK] 位置变成真实 token。
- **confidence threshold**：置信度门槛，低于门槛的位置保持掩码。
- **early stop**：块内全部揭晓后提前结束去噪。
- **KV cache**：缓存前缀的注意力键值，避免重复计算。
