# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-19

### Added

- 分块扩散推理内核：block diffusion 替代逐 token 自回归，块内并行去噪。
- 掩码调度器：cosine / linear / constant 三种 schedule。
- 置信度门槛自适应退火采样：高置信位置逐步 unmask，停滞时自动降阈值。
- 简化扩散 Transformer：embedding / attention / FFN / 扩散头，随机权重参考实现。
- 字节级 tokenizer + 简易 BPE tokenizer。
- KV cache：跨 block 复用前缀缓存（可选开关）。
- 流式输出 / 批处理 / 会话管理。
- CLI：generate / benchmark / doctor / serve / profile / inspect。
- FastAPI 服务 + Python SDK + Web 控制台。
- 对比基准：自回归 vs 扩散的吞吐 / 延迟 / 步数统计。
- Docker / docker-compose / K8s / Helm / GitHub Actions CI。
