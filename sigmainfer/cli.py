# -*- coding: utf-8 -*-
"""命令行入口：generate / benchmark / doctor / serve / profile / inspect / init。"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .model import SigmaModel, ModelConfig, build_model
from .engine import SigmaEngine, GenerationConfig
from .tokenizer import get_tokenizer
from .config_loader import load_config, dump_config
from .core import RunMetrics


def _build_engine(args):
    if args.config:
        mc, gc = load_config(args.config)
    else:
        mc, gc = ModelConfig(), GenerationConfig()
    model = SigmaModel(mc)
    return model, SigmaEngine(model, gc)


def cmd_generate(args):
    model, engine = _build_engine(args)
    tok = get_tokenizer(args.tokenizer)
    prompt_ids = tok.encode(args.prompt)
    state = engine.generate(prompt_ids)
    out = tok.decode(state.generated)
    if args.json:
        print(json.dumps({"prompt": args.prompt, "output": out,
                          "stats": state.snapshot()}, ensure_ascii=False, indent=2))
    else:
        print(out)
    return 0


def cmd_doctor(args):
    model, engine = _build_engine(args)
    tok = get_tokenizer("byte")
    checks = []
    ok = True
    # 1. 模型前向
    try:
        logits = model.forward([0, 1, 2, -1])
        assert len(logits) == 4 and len(logits[0]) == model.cfg.vocab_size
        checks.append(("model.forward", True, ""))
    except Exception as e:
        ok = False
        checks.append(("model.forward", False, str(e)))
    # 2. 生成冒烟
    try:
        st = engine.generate(tok.encode("hello"))
        assert len(st.generated) == engine.cfg.max_new_tokens
        checks.append(("generate", True, f"generated={len(st.generated)} tokens"))
    except Exception as e:
        ok = False
        checks.append(("generate", False, str(e)))
    # 3. 采样器/调度器注册
    from .core import list_samplers, list_schedules
    checks.append(("samplers", True, ",".join(list_samplers())))
    checks.append(("schedules", True, ",".join(list_schedules())))

    passed = sum(1 for _, s, _ in checks if s)
    for name, s, msg in checks:
        print(f"  [{'PASS' if s else 'FAIL'}] {name} {msg}")
    print(f"doctor: {passed}/{len(checks)} passed")
    return 0 if ok else 1


def cmd_benchmark(args):
    from .benchmark.compare import run_comparison
    model, engine = _build_engine(args)
    tok = get_tokenizer("byte")
    prompt_ids = tok.encode(args.prompt or "once upon a time")
    report = run_comparison(engine, prompt_ids, max_new_tokens=args.max_new_tokens,
                            block_size=args.block_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args):
    from .serve import run_server
    model, engine = _build_engine(args)
    run_server(engine, host=args.host, port=args.port)
    return 0


def cmd_profile(args):
    import cProfile
    import pstats
    import io
    model, engine = _build_engine(args)
    tok = get_tokenizer("byte")
    pr = cProfile.Profile()
    pr.enable()
    engine.generate(tok.encode(args.prompt or "hello world"))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(args.lines)
    print(s.getvalue())
    return 0


def cmd_inspect(args):
    model, engine = _build_engine(args)
    print(json.dumps({"model": model.cfg.as_dict(),
                      "generation": engine.cfg.as_dict()}, ensure_ascii=False, indent=2))
    return 0


def cmd_init(args):
    mc, gc = ModelConfig(), GenerationConfig()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(dump_config(mc, gc))
    print(f"已生成配置: {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sigmainfer", description="Sigma 文本扩散推理引擎")
    p.add_argument("--version", action="version", version=f"sigmainfer {__version__}")
    p.add_argument("--config", help="JSON 配置文件路径")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="生成文本")
    g.add_argument("prompt")
    g.add_argument("--tokenizer", default="byte")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("doctor", help="自检")
    d.set_defaults(func=cmd_doctor)

    b = sub.add_parser("benchmark", help="对比自回归 vs 扩散")
    b.add_argument("--prompt")
    b.add_argument("--max-new-tokens", type=int, default=64)
    b.add_argument("--block-size", type=int, default=16)
    b.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    pr = sub.add_parser("profile", help="性能剖析")
    pr.add_argument("--prompt", default="hello world")
    pr.add_argument("--lines", type=int, default=20)
    pr.set_defaults(func=cmd_profile)

    ins = sub.add_parser("inspect", help="查看当前配置")
    ins.set_defaults(func=cmd_inspect)

    i = sub.add_parser("init", help="生成默认配置")
    i.add_argument("--out", default="config.json")
    i.set_defaults(func=cmd_init)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
