#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h:h}"
cd "$EXPERIMENT_DIR"
export ALFWORLD_DATA="$EXPERIMENT_DIR/data/alfworld"
export XDG_CACHE_HOME="$EXPERIMENT_DIR/.cache"
export PYTHONPYCACHEPREFIX="$EXPERIMENT_DIR/.cache/pycache"
export PYTHONPATH="$EXPERIMENT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "$EXPERIMENT_DIR/.venv/bin/python" -u -m experience.cli "$@"
