# Reflection_Evo_Agent

基于 ALFWorld 纯文本环境的反思与经验进化实验，使用本地 Qwen3-14B Q4_K_M。模型权重保持固定，通过对比执行轨迹生成和修正经验库。

## 源码位置

```text
src/experience/       # 核心源代码：执行、反思、经验更新、抽样与评估
configs/             # 实验参数与 Qwen 模型配置
prompts/             # 执行、候选经验生成、扩展修正提示词
scripts/             # 数据准备和运行入口
tests/               # 协议测试
docs/实验流程.md       # 详细实验方法
```

## 实验设置

- 原题池 20 道，扩展池 80 道，最终测试题 20 道；固定种子 42。
- 进化 10 轮，训练题普通随机抽样、无放回；每轮从扩展池抽一道同任务结构的题。
- 每轮先固定旧经验，在同一道原题上独立运行 3 次，配对比较轨迹，生成一条候选经验。
- 候选经验只验证原题一次，失败就舍弃；通过后只运行一次扩展题，用其轨迹修正经验，然后直接追加。
- 扩展题成功不是入库条件；修正后不再复验，不做融合回归。每轮最多 5 条执行轨迹。
- 最后冻结经验库，在 20 道独立测试题上比较初始与最终经验，各执行一次，共 40 条测试轨迹。

## 安装与运行

需要 Python 3.9+ 和本地 Ollama。在 macOS 上先执行：

```sh
./安装环境.command
```

该脚本在项目内创建虚拟环境，安装文本环境并下载官方数据和分词器。模型权重、数据集、运行日志及本地虚拟环境不包含在此仓库中。

启动 Ollama 桌面应用，或在另一个终端运行适合 16GB Mac 的服务配置：

```sh
OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

在仓库根目录准备本地模型：

```sh
ollama pull qwen3:14b-q4_K_M
ollama create qwen14b-local -f configs/QwenModelfile
```

然后检查并运行：

```sh
./scripts/run.sh prepare
./scripts/run.sh check
./scripts/run.sh run --run-dir runs/main
```

macOS 也可双击 `开始实验.command`。如果本机已有相邻的 `qwen-local` 部署目录，启动入口会自动复用它；其他机器请先启动 Ollama 并创建上述模型。

按 Ctrl+C 中断，再次运行相同命令可续跑。查看进度：

```sh
./scripts/run.sh status --run-dir runs/main
```

结果保存在 `runs/main/report.md`，经验库保存在 `runs/main/memory_final.json`，完整轨迹与模型调用也会记录在运行目录中。

## 验证

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

已在 M5 / 16GB Mac 上通过 22 项流程测试，并验证 ALFWorld 环境重置和本地模型联通。尚未运行完整 10 轮实验，仓库不包含正式实验成绩。

16GB 内存余量较紧，执行上下文为 4096 token，反思上下文为 8192 token。详细参数见 `configs/default.yaml`。

Benchmark 来源：[ALFWorld（ICLR 2021）](https://github.com/alfworld/alfworld)。本实验最终测试从官方公开 `valid_unseen` 划分留出；扩展池属于开发数据。
