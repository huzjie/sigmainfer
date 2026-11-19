# SigmaInfer — Sigma 文本扩散推理引擎

> 用**分块扩散（block diffusion）**替代逐 token 自回归，把「一次生成一个词」改成
> 「一块并行生成几十个词」。纯 Python 零依赖内核，填一份 JSON 配置即可跑。
> 对应英伟达开源的 Nemotron-Labs-Sigma（3B/8B）：分块扩散 + 32 步去噪 + 64 token 分块，
> H100 上跑到 120~140 token/s。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 为什么会有它（一段背景，引出技术）

大模型的文本生成几乎默认是**自回归（autoregressive）**：按顺序一次预测下一个 token，
`t1 → t2 → t3 …`。这带来两个硬伤——**速度慢**（N 个 token 就要 N 次串行前向，没法
并行）和**错误会级联**（前面的 token 错了，后面一路错下去）。

英伟达 Nemotron-Labs-Sigma 换了一条路：**把「生成」当「去噪」来做**。它借鉴图像扩散
（把一张纯噪声图逐步去噪成清晰图）的思路，把一段文本切成若干块，每块内所有位置先
打上 [MASK]，再通过若干步「去噪」逐步把高置信度的位置从 [MASK] 变成真实 token，
直到整块成型。因为**块内所有位置可以并行预测**，吞吐量成倍提升；H100 单卡 BF16
下跑到 120~140 token/s。

SigmaInfer 就是把这套「分块扩散」落成可运行、可拆解、可复用的工程：一块 YAML/JSON
配置声明「块多大、去噪几步、用哪种采样、置信度门槛多高」，引擎照做，并给你吞吐、
步数、早停次数等全部可观测指标。

---

## 核心能力

| 模块 | 做什么 |
| --- | --- |
| 分块调度 | 把 `max_new_tokens` 切成若干 block，块内并行、块间串行 |
| 掩码调度 | cosine / linear / constant 三种 schedule 控制每步 unmask 比例 |
| 置信度采样 | confidence / topk / argmax 三种采样器 + 门槛过滤 |
| 自适应退火 | 停滞自动降门槛，保证每块「一定出结果」 |
| 扩散 Transformer | 随机权重参考实现（embedding/注意力/FFN/扩散头，可换真实权重） |
| 多轮会话 | Session / SessionManager，历史拼接做上下文 |
| 对比基准 | 同一引擎下自回归 vs 扩散的吞吐 / 步数 / 延迟对比 |
| 工程交付 | CLI + FastAPI + Python SDK + Web 控制台 + Docker/K8s/Helm/CI |

---

## 快速开始（填配置即可跑）

```bash
# 1. 安装（核心零依赖）
pip install -e .

# 2. 生成一份默认配置
sigmainfer init --out config.json

# 3. 自检
sigmainfer doctor

# 4. 生成文本
sigmainfer generate "今天想聊聊" --json

# 5. 对比自回归 vs 扩散
sigmainfer benchmark --max-new-tokens 64 --block-size 16

# 6. 起服务
pip install -e ".[serve]"
sigmainfer serve --port 8000   # 打开 http://127.0.0.1:8000/docs
```

> 说明：仓库默认用**随机初始化的缩小规模权重**（dim=64 / 2 层 / 字节词表 256），
> 目的是让你在任意机器上把「分块扩散」整条链路跑通、看指标、改参数；换真实
> Sigma 3B/8B 权重只需替换 `sigmainfer/model/` 的实现（见 `docs/architecture.md`）。

---

## 技术原理拆解（这篇的重点）

### 1. 生成范式：自回归 vs 扩散，差在哪

自回归是**串行**的：每个 token 都要等前面的 token 算完。扩散是**并行**的：把一整块
token 当作「待去噪的图像」，一次前向对块内所有 [MASK] 位置同时打分，再批量 unmask。

```
自回归:  t1 -> t2 -> t3 -> ... -> tN      （N 次串行前向）
扩散:    [M] [M] [M] ... [M]              （块内 N 个位置并行）
         第1步去噪 -> 部分 unmask
         第2步去噪 -> 更多 unmask
         ... 直到整块成型
```

**关键决策**：块多大是吞吐和质量的权衡。块越大并行度越高、吞吐越好，但单块内「一次
决定太多位置」会让低置信位置被强行定死，质量下降。Sigma 选了 64 token 一块，配合
32 步去噪，是工程上的甜点区。

### 2. 掩码调度（mask schedule）：每步「揭开」多少

一个 block 从全掩码出发，不能一步全揭开（那样就是一次 argmax，退化成质量很差的
并行贪心）。要像图像扩散一样**逐步去噪**。每步保留多少掩码由 schedule 决定：

```python
# cosine：早期大步 unmask、后期精细收敛（MaskGIT 风格）
keep = cos(pi/2 * step/total_steps)     # step=0 -> 1.0，step=T -> 0.0
# linear：每步均匀揭开 1/T
keep = 1 - step/total_steps
```

对应代码在 `sigmainfer/core/schedule.py`，`cosine/linear/constant` 三种可换。

### 3. 置信度门槛 + 自适应退火：让「早停」成为可能

如果每步固定 unmask 全部高置信位置，一个 64 token 的块可能 3 步就解完，也可能 8 步
还卡着几个位置。核心技巧是**置信度门槛 + 退火**：

- 每步只 unmask `置信度 >= threshold` 的位置（默认 0.7）；
- 若连续 `stall_steps` 步 unmask 数为 0，说明门槛太严，**自动下调** `threshold *= decay`；
- 步数耗尽仍有余留，**强制 argmax 收尾**——保证每块一定产出结果，不留死块。

```python
from sigmainfer.core import ConfidenceThreshold
th = ConfidenceThreshold(initial=0.7, min_value=0.2, decay=0.9, stall_steps=2)
keep_p, keep_v, _ = th.filter(positions, values, confidences)  # 过滤低置信
th.update(num_unmasked, num_masked_before)                     # 退火
```

### 4. 采样器：confidence / topk / argmax 的取舍

- **confidence**：每个掩码位置取 argmax，置信度 = softmax 概率，交给门槛统一过滤。
- **topk**：每步只对置信度最高的 k 个位置做决策，其余保持掩码（更激进、更快）。
- **argmax**：无温度确定性输出，用于可复现基准。

三者共享同一接口 `sample(logits, step, rng) -> (positions, values, confidence)`，
在 `sigmainfer/core/sampler.py` 里可替换。

### 5. KV Cache：跨块复用前缀

已生成的 block 是后续 block 的前缀。不缓存的话，每个新 block 都要把前缀重新算一遍
注意力，计算量随长度平方膨胀。`sigmainfer/core/kv_cache.py` 提供按 `(layer, position)`
存储 key/value 的参考实现——新块只算增量位置的注意力。

---

## 可直接复用的「干货块」

### 干货块 A：分块扩散的伪代码工作流（照着抄）

```
输入: prompt_ids, max_new_tokens, block_size, num_steps
prefix = prompt_ids
while 还剩要生成的 token:
    block = [MASK] * min(block_size, 剩余)      # 1. 开一个新块，全掩码
    threshold.reset()
    for step in 1..num_steps:                    # 2. 块内去噪循环
        full = prefix + block
        logits = model(full)                      # 一次前向，全位置打分
        cand = sampler.sample(logits[masked])     # 3. 掩码位置候选 + 置信度
        keep = cand[conf >= threshold]            # 4. 门槛过滤
        block.unmask(keep)                        # 5. 高置信位置揭晓
        threshold.anneal(keep.empty)              # 6. 停滞则降门槛
        if block 全部揭晓: break                  # 7. 早停
    block 剩余位置 argmax 强制收尾                 # 8. 保证出结果
    prefix += block
```

### 干货块 B：置信度门槛退火的 4 个参数怎么调

| 参数 | 默认 | 调大 | 调小 |
| --- | --- | --- | --- |
| `initial` | 0.7 | 更保守、质量更高、步数更多 | 更快、质量略降 |
| `min_value` | 0.2 | 允许更低的兜底门槛 | 更早卡住 |
| `decay` | 0.9 | 停滞时降门槛更慢 | 降得快、易出结果 |
| `stall_steps` | 2 | 容忍更多停滞才退火 | 更快触发退火 |

### 干货块 C：一句话踩坑清单

- 块太大 ≠ 更好：并行度上去了，但低置信位置被强行定死，质量塌方。**先固定块大小跑
  吞吐，再回头调门槛保质量**。
- 全掩码块一步 argmax = 并行贪心，不是扩散。**必须多步去噪**才有质量。
- 早停必须配**兜底收尾**，否则一个块卡死几个位置，整条生成就挂了。
- 门槛退火要**设下限**（min_value），否则门槛一路降到 0，退化成无门槛。
- 基准对比要**同引擎、同权重、同 prompt**，否则自回归 vs 扩散的差距被实现噪声淹没。

---

## 目录结构

```
sigmainfer/
├── sigmainfer/          # 核心包（零依赖）
│   ├── core/            #   令牌 / 块 / 状态 / 调度 / 采样 / 门槛 / KV cache
│   ├── model/           #   扩散 Transformer（embedding/注意力/FFN/头/权重）
│   ├── engine/          #   推理 / 流式 / 批处理 / 会话 / 块调度
│   ├── tokenizer/       #   字节级 + 字符级 tokenizer
│   ├── benchmark/       #   自回归 vs 扩散对比
│   ├── sdk/             #   Python 客户端
│   ├── cli.py           #   generate/benchmark/doctor/serve/profile/inspect/init
│   └── serve.py         #   FastAPI
├── config/              # 示例配置（JSON）
├── examples/            # 快速开始脚本
├── prompts/             # 可复用提示词 / 配置模板
├── deploy/              # Docker / compose / K8s / Helm / CI
├── ts-sdk/              # TypeScript 客户端
├── tools/               # 评测 / 脚本
├── tests/               # 单元测试
└── docs/                # 架构 / 扩散原理 / 采样 / 基准 / 踩坑
```

## License

Apache 2.0。本仓库为**参考实现**，随机权重仅用于演示与基准，非生产语言模型；
加载真实 Sigma 权重时请校验文件 SHA-256。
