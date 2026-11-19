# 故障排查

## 生成极慢

调小 dim / n_layers（参考实现是纯 Python 矩阵运算），或调小 max_new_tokens。

## 输出全是乱码

随机权重无语义，属预期。换真实权重才有可读文本。

## doctor 报 generate 失败

检查 config.json 字段是否与 GenerationConfig 参数对齐。
