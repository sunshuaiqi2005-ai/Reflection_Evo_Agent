# Reflection_Evo_Agent

反思与经验进化实验代码。

完整 ALFWorld 实验项目位于 **[src/alfworld-experience/](src/alfworld-experience/)**，包含源码、配置、提示词、脚本、测试和使用文档。

```text
src/
└── alfworld-experience/
    ├── src/experience/     # 核心 Python 源码
    ├── configs/
    ├── prompts/
    ├── scripts/
    ├── tests/
    ├── docs/
    └── README.md
```

从仓库根目录进入实验项目后，按照其 README 安装和运行：

```sh
cd src/alfworld-experience
./安装环境.command
```

实验使用本地 Qwen3-14B 4bit；40 道训练题、48 道扩展题、20 道最终测试题，均为清洗、加热、冷却后放置任务，按训练清单顺序进化 40 轮。详细流程与启动方法见 **[项目说明](src/alfworld-experience/README.md)**。
