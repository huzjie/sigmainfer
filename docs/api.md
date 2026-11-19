# API 参考

## POST /v1/generate

请求：
```json
{"prompt": "hello", "max_new_tokens": 64, "block_size": 16, "num_steps": 8}
```

响应：
```json
{"output": "...", "stats": {"generated_len": 64, "total_blocks": 4}}
```

## GET /health

```json
{"status": "ok"}
```
