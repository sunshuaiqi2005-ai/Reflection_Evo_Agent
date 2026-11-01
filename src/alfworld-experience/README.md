# ALFWorld 反思与经验进化

基于 ALFWorld 纯文本环境的反思与经验进化实验，使用本地 Qwen3-14B Q4_K_M。模型权重保持固定，通过对比执行轨迹生成和修正经验库。

从仓库根目录先进入本项目：

```sh
cd src/alfworld-experience
```

下文命令均在本项目目录执行。

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

- 原题池 20 道，扩展池 68 道（目标 80，可用 68），最终测试题 20 道；固定种子 42。
- 训练、扩展和测试均只选清洗、加热、冷却后放置三类任务，作为本实验的中等任务范围（并非官方难度标签）。训练从 1642 道同类题中普通随机抽 20 道；扩展池只取官方 valid_seen 的 68 道可用同类题；测试从 valid_unseen 的 75 道同类题中普通随机抽 20 道。
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

在本项目目录准备本地模型：

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

已在 M5 / 16GB Mac 上通过 42 项检查，包含真实环境任务完成、三类题池筛选和同结构配对检查。开发题预检不计入正式实验成绩。

16GB 内存余量较紧，执行上下文为 4096 token，反思上下文为 8192 token。详细参数见 `configs/default.yaml`。

Benchmark 来源：[ALFWorld（ICLR 2021）](https://github.com/alfworld/alfworld)。本实验最终测试从官方公开 `valid_unseen` 划分留出；扩展池属于开发数据。

## 全局提示和环境协议

全局提示位于 `prompts/actor.txt`。TextWorld 放置使用 `move OBJECT to RECEPTACLE`，并明确手持限制、位置、容器开关和失败反馈恢复；执行输入还包含仅由本条轨迹可见反馈整理的状态笔记。旧 `put ... in/on ...` 别名会等价翻译，原始输出和标记保留。

```sh
# 完整开发题预检（不计入正式结果）
./scripts/run.sh pilot
# 真实环境回归检查（先安装数据）
ALFWORLD_INTEGRATION=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

旧错误语法运行与简单任务运行分别归档；中等任务实验使用空经验库和新的运行目录，不混用旧结果。

三个题池分别由 `task_selection.train_task_types`、`extension_task_types` 和 `test_task_types` 筛选，当前均固定为三类处理后放置任务。测试清单在执行前固定，与训练、扩展题互不重叠。只调整最终测试筛选时，会保留原始训练元数据并单独记录 `evaluation_metadata.json`；训练题、种子、模型或执行/学习代码改变仍不能混用旧运行结果。


## 反思输入预算

反思上下文为 8192 token，预留输出 1536 token。采用无损紧凑格式：相同任务类型及初始观察在 `contexts` 中仅保存一次；每条轨迹引用自己的上下文；`step_fields` 一次声明步号、动作、反馈、得分和成功标记，各步用数组保存。原始轨迹不变，所有步骤和反馈均保留；不同原题、扩展题的初始观察分别保存。

使用旧运行中两组各含两条 50 步失败轨迹的开发配对测得（空经验库，包含完整提示模板及安全预留）：

| 开发轮次 | 原输入 token | 紧凑输入 token | 减少 | 保留 1536 输出后的余量 |
| --- | --- | --- | --- | --- |
| 1 | 7424 | 5008 | 32.5% | 1648 |
| 2 | 7199 | 4812 | 33.2% | 1844 |

该优化减少格式和共同信息的重复占用，模型上下文上限仍为 8192。预算检查继续在每次调用前执行，以上测量不代表任意后续经验库和轨迹都一定能容纳。轨迹引用仍使用原始步号，证据需与原始动作及反馈精确匹配。
