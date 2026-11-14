# V2：问题预判与聚合经验自改进

固定模型权重，使用 Planner、Actor、Problem Forecaster 和 Critic 四份提示词完成学习。同一道训练题由 Planner 生成三个方案，Actor 分别重置环境执行，Critic 比较执行记录后生成经验；整道题只提交一次经验库更新。

## 运行

本机 `assets` 和 `.venv` 链接到相邻 `alfworld-experience` 的数据及环境，无需重复安装。先启动 `../qwen-local/启动服务.command`，然后执行：

```sh
./scripts/run.sh check
./scripts/run.sh smoke
./scripts/run.sh train --run-dir runs/main
./scripts/run.sh evaluate --run-dir runs/main
# 或依次训练、冻结测评
./scripts/run.sh run --run-dir runs/main
```

`smoke` 只执行一题、三方案、每方案最多两步，用于联通诊断，结果不能作为正式实验成绩。可使用 `--limit 2 --max-steps 5 --run-dir runs/pilot` 运行独立预检。

```sh
./scripts/run.sh status --run-dir runs/main
./scripts/run.sh report --run-dir runs/main
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

换机器时安装本项目，在配置中指定官方 ALFWorld 数据和 Qwen tokenizer 路径，并准备 Ollama 模型：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
ollama pull qwen3:14b-q4_K_M
ollama pull embeddinggemma
```

本机模型名称是 `qwen14b-local`；其他机器可在 `configs/default.yaml` 改为实际安装的模型名。`assets/alfworld` 应指向已解压的官方文本数据，`assets/tokenizer/tokenizer.json` 为 Qwen3-14B 的 tokenizer。模型服务和数据准备沿用原项目，不在 v2 中重复下载数据。

## 每题流程

1. 固定经验库快照。规划经验少于 `memory.full_threshold` 时全部注入；达到阈值后，Forecaster 参考规划形态判断经验及全部规划目录，选择相关项目。
2. 注入选中项目的全部经验，Planner 一次生成三个有策略差异的方案。
3. 每个方案执行前，根据题目、当前方案和初始观察，同样选择并注入执行经验。每条完整轨迹只注入一次，执行中 Actor 持续读取环境反馈，不再次调用 Forecaster。
4. 三个方案均执行完后，Critic 输出判断、规划、执行三类经验。判断经验含 planning 和 execution 两种形态，共四个逻辑集合。通常合计 3—8 条，数量只是提示目标。
5. context 和 problem 分别 embedding，归一化后等权拼接。四个集合分别与簇中心比较，达到相似度阈值就加入，否则新建簇；簇数和问题名称由数据产生。
6. Critic 批量整理本题改变的簇，合并重复内容，保留不同方法与条件。形成新的目录和经验条目后，一次写入本题结果。

只有目录选择参与读取；embedding 用于写入归簇。达到阈值后也不限制选中项目内的条目数量，不进行条目级 Top-k。判断者可选择零个或多个项目。

## 经验结构

```json
{
  "id": "E00001",
  "type": "judge",
  "phase": "execution",
  "context": "适用的题目、方案或状态条件",
  "problem": "自由描述的问题",
  "explanation": "可以为空，也可以保留不确定解释",
  "methods": ["方法一", "方法二"],
  "advice": ["对应建议"],
  "exceptions": [],
  "cluster_id": "C00001"
}
```

`type` 为 judge / planner / actor；judge 通过 phase 区分两种形态。经验正文不包含证据、来源轨迹和来源步骤字段。Planner、Actor 的 phase 分别为 planning、execution，仅表示使用阶段。

簇保存 ID、用途、阶段、state_name、problem_name、scope、experience_ids 和 embedding 中心。Forecaster 只看相应角色的目录名称与适用条件，程序根据返回 ID 读取全部条目。

## 数据与测评

`data/tasks.json` 复制了既有固定清单：40 道训练、48 道开发、20 道测试。数据清单包含环境文件路径；模型只接收环境返回的可见任务与观察，不接收题目路径、专家计划或合法动作列表。训练阶段不加载测试题作为经验来源，不进行原题或扩展题注入验证。

学习阶段最多 40×3=120 条执行轨迹。冻结评测对空经验、最终经验分别生成并执行三个方案，20 题每组 60 条轨迹。两组使用相同环境和采样种子；报告执行成功率及三方案至少一次成功的题目比例，避免把三方案覆盖率当成单次成功率。评测不调用 Critic、不更新经验。

## 文件

- `src/forecast_agent/`：模型接口、四角色、经验库、环境和实验入口。
- `prompts/`：角色及聚合提示词。
- `configs/default.yaml`：模型、上下文、经验阈值和步数配置。
- `runs/<name>/rounds/`：每题一次提交的结果，包含该题结束后的经验快照。
- `runs/<name>/episodes/`：完整动作及反馈，注入经验固定在每条轨迹开头。
- `runs/<name>/model_calls/`：模型输入输出缓存。
- `runs/<name>/events.jsonl`：调用、注入、动作与更新记录。
- `runs/<name>/memory_final.json`：冻结经验；`report.md` / `report.json`：汇总结果。

再次使用同一配置和目录会复用已完成题目、轨迹及模型调用。更改实验配置时使用新目录。推理错误直接抛出；经验不会被自动过滤、改写命令或静默截断。历史只按 `episode.history_tokens` 保留最近窗口，完整轨迹始终保存。超过上下文时明确报错，可调整角色上下文配置或通过经验聚合缩短内容。

GitHub 同步目录为现有仓库根目录的 `v2/`；本地虚拟环境、模型、官方数据、缓存和运行产物不上传。
