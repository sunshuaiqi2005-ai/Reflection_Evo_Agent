# V2：问题预判与状态索引经验自改进

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

本次参数和提示词更新后使用新的运行目录，旧运行结果保持原样；例如用 `./scripts/run.sh smoke --run-dir runs/smoke-budget12` 重新检查。

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

1. 固定经验库快照。当前角色的经验少于 `memory.full_threshold`（默认 8 条）时，全库作为候选；达到阈值后，Forecaster 参考相应形态的判断经验及全部状态索引目录，按相关性从高到低选择并排列状态 ID。
2. 按 `memory.max_injected_experiences`（默认 12 条）的每次注入上限选择候选经验，Planner 一次生成三个有策略差异、围绕同一本题目标的方案。
3. 每个方案执行前，根据题目、当前方案和初始观察，同样选择并注入执行经验。每条完整轨迹只注入一次，执行中 Actor 持续读取环境反馈，不再次调用 Forecaster。
4. 三个方案均执行完后，Critic 在同一次调用中先核对本题目标及完成条件，再对照实际动作、环境反馈和 `won`，输出判断、规划、执行三类经验。判断经验含 planning 和 execution 两种形态，共四个逻辑集合。通常合计 8—12 条，数量只是提示目标；有更多有价值的经验时可以超出，不要求凑数，也不固定各类配额。
5. 只对 `state` 做 embedding 并归一化。四个集合分别与簇中心比较，达到 `memory.similarity_threshold`（默认 0.60）就加入，否则新建状态索引；簇数和状态描述由经验产生。相似度阈值由 0.80 降到 0.60 后，相近状态更容易归入同一簇。
6. 同一状态索引下保留各条经验及原有 ID，解释、具体做法、建议和例外不参与归簇，也不交给 LLM 重写。目录描述选取簇内最接近向量中心的成员 `state`，再一次写入本题结果。

Planner 和 Actor 的输入单独列出 `task_goal`，直接取自环境初始观察中的原始任务文本，同时保留完整初始观察。当前任务要求的对象、属性和目标位置作为本题约束；经验是有适用条件的建议，不能替换这些要求或当前观察事实。

Critic 区分发出动作与动作实际完成，依据环境反馈及 `won` 判断结果。`step_limit` 记录终止原因，不能直接当作失败根因；没有反馈支持的原因保留为条件性解释或尚未验证的推测。

只有状态目录选择参与读取；embedding 用于写入归簇和状态索引归并。判断者可选择零个或多个状态，其自身相应形态的判断经验输入保持原有方式；12 条上限只作用于注入 Planner 或 Actor 的经验。

选中状态下的候选经验总数不超过 12 条时，全部注入。超过 12 条时，按 Forecaster 返回的状态顺序轮询：每轮从各状态按已有成员顺序取一条，取完某状态就略过，直到选满 12 条。这是此前整簇注入规则的超额例外；不增加 LLM 调用、embedding、正文重排或过滤，也不删除经验库条目。全库候选分支同样遵守每次注入上限，默认少于 8 条时会全部注入。

注入日志记录 `candidate_count`、`budget_applied` 和实际注入的经验 ID，区分判断者选中的状态与最终提供给模型的条目。12 条是数量上限，完整上下文仍按 token 预算检查。

## 经验结构

```json
{
  "id": "E00001",
  "type": "judge",
  "phase": "execution",
  "state_id": "S00001",
  "state": "适用背景，以及出现的问题或所处状态",
  "explanation": "可以为空，也可以保留不确定解释",
  "advice": ["具体解决办法、不同策略和使用建议"],
  "exceptions": []
}
```

`type` 为 judge / planner / actor；judge 通过 phase 区分两种形态。经验正文不包含证据、来源轨迹和来源步骤字段。Planner、Actor 的 phase 分别为 planning、execution，仅表示使用阶段。

状态索引保存 `state_id`、用途、阶段、代表 `state`、`experience_ids` 和 embedding 中心。Forecaster 看到的待选目录只有相应角色的 `{state_id, state}`，程序根据返回 ID 收集全部候选条目，再按上述注入预算选择。每条经验保留自己的 `state` 原文；具体做法与建议统一存入 `advice`。

## 数据与测评

`data/tasks.json` 保留原 40 道训练题，按原顺序将开发（验证）清单的前 10 道移入训练，组成固定 50 道训练题；开发清单剩余 38 道，测试仍为原 20 道，三个清单互不重叠。题目原始 ID 和 `official_split` 保留。数据清单包含环境文件路径；模型只接收环境返回的可见任务与观察，不接收题目路径、专家计划或合法动作列表。训练阶段不加载测试题作为经验来源，不进行原题或扩展题注入验证。

学习阶段最多 50×3=150 条执行轨迹。冻结评测对空经验、最终经验分别生成并执行三个方案，20 题每组 60 条轨迹。两组使用相同环境和采样种子；报告执行成功率及三方案至少一次成功的题目比例，避免把三方案覆盖率当成单次成功率。评测不调用 Critic、不更新经验。

## 文件

- `src/forecast_agent/`：模型接口、四角色、经验库、环境和实验入口。
- `prompts/`：四个角色的提示词。
- `configs/default.yaml`：模型、上下文、经验阈值和步数配置。
- `runs/<name>/rounds/`：每题一次提交的结果，包含该题结束后的经验快照。
- `runs/<name>/episodes/`：完整动作及反馈，注入经验固定在每条轨迹开头。
- `runs/<name>/model_calls/`：模型输入输出缓存。
- `runs/<name>/events.jsonl`：调用、注入、动作与更新记录。
- `runs/<name>/memory_final.json`：冻结经验；`report.md` / `report.json`：汇总结果。

再次使用同一配置和目录会复用已完成题目、轨迹及模型调用。更改实验配置或经验字段时使用新目录。推理错误直接抛出；经验条目按注入预算选择并记录，正文保持原样，动作不会被自动改写。Actor 历史按 `episode.history_tokens`（默认由 1600 调整为 3200 token）保留最近窗口，完整轨迹始终保存。Planner、Actor、Forecaster 的总上下文仍为 8192，Critic 仍为 16384，各角色输出预算保持现有配置。超过上下文时明确报错，可调整角色上下文配置。

GitHub 同步目录为现有仓库根目录的 `v2/`；本地虚拟环境、模型、官方数据、缓存和运行产物不上传。
