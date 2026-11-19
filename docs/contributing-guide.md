# 贡献指南（文档版）

1. Fork 仓库，clone 到本地。
2. `pip install -e ".[dev]"` 安装开发依赖。
3. 新采样器：实现 `BaseSampler`，在 `core/registry.py` 注册。
4. 新 schedule：实现 `keep_fraction`，在 `core/schedule.py` 注册。
5. 提交前跑 `python -m pytest tests -q` 与 `sigmainfer doctor`。
