#!/usr/bin/env bash
set -euo pipefail
pip install -e .
python -m compileall sigmainfer
echo "build ok"
