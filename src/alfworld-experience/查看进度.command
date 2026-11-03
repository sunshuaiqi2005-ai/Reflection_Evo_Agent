#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
LATEST_RUN_DIR="$("$EXPERIMENT_DIR/.venv/bin/python" - "$EXPERIMENT_DIR" <<'PY_RUN'
from pathlib import Path
import sys
root = Path(sys.argv[1])
statuses = list((root / "runs").glob("*/status.json"))
print(max(statuses, key=lambda p: p.stat().st_mtime).parent if statuses else root / "runs/main")
PY_RUN
)"
"$EXPERIMENT_DIR/scripts/run.sh" status --run-dir "$LATEST_RUN_DIR"
print '\n按回车关闭。'
read -r
