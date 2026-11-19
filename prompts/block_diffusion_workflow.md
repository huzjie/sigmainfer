# 分块扩散伪代码工作流（可复用到任意框架）

```
输入: prompt_ids, max_new_tokens, block_size, num_steps
prefix = prompt_ids
while 还剩要生成的 token:
    block = [MASK] * min(block_size, 剩余)      # 1. 开新块，全掩码
    threshold.reset()
    for step in 1..num_steps:                    # 2. 块内去噪循环
        full = prefix + block
        logits = model(full)                      # 一次前向，全位置打分
        cand = sampler.sample(logits[masked])     # 3. 掩码候选 + 置信度
        keep = cand[conf >= threshold]            # 4. 门槛过滤
        block.unmask(keep)                        # 5. 高置信位置揭晓
        threshold.anneal(keep.empty)              # 6. 停滞则降门槛
        if block 全部揭晓: break                  # 7. 早停
    block 剩余位置 argmax 强制收尾                 # 8. 保证出结果
    prefix += block
```

要点：步骤 2 是「并行」的关键——一次前向对块内所有 [MASK] 位置同时打分；步骤 6 的
退火和步骤 8 的兜底是「工程可用」的关键，缺一个都会卡死或退化成并行贪心。
