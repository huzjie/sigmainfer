# -*- coding: utf-8 -*-
"""异常类型。"""


class SigmaInferError(Exception):
    """基础异常。"""


class ConfigError(SigmaInferError):
    """配置非法。"""


class ModelError(SigmaInferError):
    """模型相关错误。"""


class GenerationError(SigmaInferError):
    """生成过程错误。"""
