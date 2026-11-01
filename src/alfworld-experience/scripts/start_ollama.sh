#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h:h}"
if /usr/bin/curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  exit 0
fi
for QWEN_DEPLOY_DIR in "$EXPERIMENT_DIR/../qwen-local" "$EXPERIMENT_DIR/../../../qwen-local"; do
  if [[ -x "$QWEN_DEPLOY_DIR/启动服务.command" ]]; then
    exec "$QWEN_DEPLOY_DIR/启动服务.command"
  fi
done
print -u2 '请先启动 Ollama，并按 README 创建 qwen14b-local 模型，然后重新运行。'
exit 1
