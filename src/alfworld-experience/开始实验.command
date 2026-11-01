#!/bin/zsh
set -eu
EXPERIMENT_DIR="${0:A:h}"
"$EXPERIMENT_DIR/scripts/start_ollama.sh"
print 'ALFWorld：20 道原题池 / 80 道扩展池 / 20 道测试题；进化 10 轮。'
print '训练题随机抽取；候选原题验证一次，扩展反馈一次，修正后直接追加。'
print '按 Ctrl+C 中断；再次运行本文件可续跑。'
exec /usr/bin/caffeinate -i "$EXPERIMENT_DIR/scripts/run.sh" run --run-dir runs/main
