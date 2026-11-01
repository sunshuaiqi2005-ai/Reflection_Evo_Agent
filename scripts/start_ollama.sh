#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h:h}"
if /usr/bin/curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  exit 0
fi
if [[ -x "$EXPERIMENT_DIR/../qwen-local/启动服务.command" ]]; then
  exec "$EXPERIMENT_DIR/../qwen-local/启动服务.command"
fi
print -u2 '请先启动 Ollama，并按 README 创建 qwen14b-local 模型，然后重新运行。'
exit 1
