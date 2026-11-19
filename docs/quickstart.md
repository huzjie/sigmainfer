# 快速开始

```bash
pip install -e .
sigmainfer init --out config.json
sigmainfer doctor
sigmainfer generate "你好" --json
sigmainfer benchmark --max-new-tokens 64 --block-size 16
```

5 条命令即可跑通整条链路。详见 README。
