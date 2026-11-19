# FAQ

## 为什么核心零依赖？

方便在任意环境（CI、容器、跳板机）直接跑，不引入供应链风险。

## 和英伟达 Sigma 什么关系？

SigmaInfer 是 Sigma「分块扩散」思想的参考实现：把 block diffusion + 掩码调度 +
置信度门槛这条链路做成可运行、可拆解、可复用的工程。默认用随机缩小规模权重跑通
流程，换真实 Sigma 权重只需替换 model 层。

## 能直接用吗？

能跑通完整链路、看指标、改参数、对比基准。但随机权重生成的文本无语义，属预期行为。

## 自回归和扩散能混用吗？

能。`benchmark/compare.py` 就是同一引擎下切换 block_size 实现两者对比。

## 怎么换真实权重？

实现 `sigmainfer/model/sigma_model.py` 里的权重加载，从 safetensors 读入并映射到
`Embedding / TransformerBlock / DiffusionHead` 的张量结构即可。
