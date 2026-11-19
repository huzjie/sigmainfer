# -*- coding: utf-8 -*-
"""工具：一键自检。"""
from sigmainfer.cli import cmd_doctor
import sys

sys.exit(cmd_doctor(__import__("argparse").Namespace(config=None)))
