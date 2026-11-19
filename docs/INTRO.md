# SigmaInfer：把「生成」当「去噪」——分块扩散并行解码的技术拆解

## 一句话背景

大模型的文本生成默认是自回归：`t1 → t2 → t3 …` 一次一个，N 个 token 就要 N 次串行
前向，慢，而且前面的错会一路级联。英伟达开源的 Nemotron-Labs-Sigma（3B/8B）换了一条
路：**把生成当去噪**。把一段文本切成块，块内所有位置打上 [MASK]，再通过若干步「去噪」
逐步把高置信位置从 [MASK] 变成真实 token，整块并行成型，H100 上跑到 120~140 token/s。

SigmaInfer 把这套「分块扩散」落成可运行、可拆解、可复用的工程，纯 Python 零依赖，
填配置即可跑。

## 技术原理拆解

**1. 生成范式**——自回归是串行的（每 token 等前面），扩散是并行的（块内一次前向对
所有 [MASK] 位置同时打分再批量 unmask）。块大小是吞吐与质量的权衡：块越大并行越高，
但低置信位置被强行定死会伤质量。

**2. 掩码调度**——块不能一步全揭开（那退化成并行贪心），要逐步去噪。每步保留多少
掩码由 schedule 决定：cosine（早期大步、后期精细）、linear（均匀）、constant（固定比例）。

**3. 置信度门槛 + 退火**——每步只 unmask `置信度 >= threshold` 的位置；连续停滞则
自动降门槛；步数耗尽则 argmax 强制收尾，保证每块一定出结果。

**4. 采样器**——confidence（argmax+概率门槛）、topk（每步只揭最高 k 个）、argmax
（确定性）。三者共享同一接口，可替换。

**5. KV Cache**——已生成 block 是后续块的前缀，缓存其 key/value，新块只算增量注意力，
避免计算量平方膨胀。

## 可复用要点（照着做）

- 分块扩散的伪代码工作流见 README「干货块 A」，可直接照抄到任意框架。
- 门槛退火 4 参数（initial/min_value/decay/stall_steps）的调参表见 README「干货块 B」。
- 三个坑：块太大伤质量、一步 argmax 不是扩散、早停必须配兜底收尾。

## 快速开始

```bash
pip install -e .
sigmainfer doctor
sigmainfer generate "今天想聊聊" --json
sigmainfer benchmark --max-new-tokens 64 --block-size 16
```

详见仓库 README 与 `docs/`。
