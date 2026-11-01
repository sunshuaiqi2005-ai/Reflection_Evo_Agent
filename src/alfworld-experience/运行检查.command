#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
"$EXPERIMENT_DIR/scripts/start_ollama.sh"
"$EXPERIMENT_DIR/scripts/run.sh" check
"$EXPERIMENT_DIR/scripts/run.sh" smoke
"$EXPERIMENT_DIR/scripts/run.sh" pilot
print '\n检查完成；这不是完整实验结果。按回车关闭。'
read -r
