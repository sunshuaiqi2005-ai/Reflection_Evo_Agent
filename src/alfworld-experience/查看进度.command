#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
"$EXPERIMENT_DIR/scripts/run.sh" status --run-dir runs/main
print '\n按回车关闭。'
read -r
