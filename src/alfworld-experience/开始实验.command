#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
"$EXPERIMENT_DIR/scripts/start_ollama.sh"
print 'ALFWorld：40 道训练题 / 48 道扩展题 / 20 道测试题；进化 40 轮。'
print '训练、扩展和测试使用清洗、加热、冷却后放置任务。'
print '先评测20题基线，再顺序训练40轮，最后评测同一组20题。'
print '按 Ctrl+C 中断；再次运行本文件可续跑。'
exec /usr/bin/caffeinate -i "$EXPERIMENT_DIR/scripts/run.sh" run --run-dir runs/main
