# -*- coding: utf-8 -*-
"""采样器 / 调度器 / 模型 的注册表。"""
from __future__ import annotations

from typing import Callable, Dict

SAMPLER_REGISTRY: Dict[str, Callable] = {}
SCHEDULE_REGISTRY: Dict[str, Callable] = {}
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_sampler(name: str):
    def deco(fn):
        SAMPLER_REGISTRY[name] = fn
        return fn
    return deco


def register_schedule(name: str):
    def deco(fn):
        SCHEDULE_REGISTRY[name] = fn
        return fn
    return deco


def register_model(name: str):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def list_samplers() -> list:
    return sorted(SAMPLER_REGISTRY)


def list_schedules() -> list:
    return sorted(SCHEDULE_REGISTRY)


def list_models() -> list:
    return sorted(MODEL_REGISTRY)
