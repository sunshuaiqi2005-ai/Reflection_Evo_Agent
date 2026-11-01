#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
cd "$EXPERIMENT_DIR"
export ALFWORLD_DATA="$EXPERIMENT_DIR/data/alfworld"
export XDG_CACHE_HOME="$EXPERIMENT_DIR/.cache"
export PYTHONPYCACHEPREFIX="$EXPERIMENT_DIR/.cache/pycache"
export PIP_CACHE_DIR="$EXPERIMENT_DIR/.cache/pip"
export PATH="$EXPERIMENT_DIR/.venv/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=4
mkdir -p logs downloads
if [[ ! -x .venv/bin/python ]]; then
  /usr/bin/python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel 'cmake>=3.26,<4'
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/prepare_assets.py
.venv/bin/python -m pip freeze > requirements.lock.txt
scripts/run.sh prepare
print '安装完成。双击“开始实验.command”可运行完整实验。'
