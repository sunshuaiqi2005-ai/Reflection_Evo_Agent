from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import sys

import yaml

from .client import OllamaClient
from .cognition import Cognition, object_schema
from .data import prepare
from .episode import AlfworldEnvironment, EpisodeRunner
from .memory import Memory
from .pipeline import Evolution, evaluate
from .util import digest, event, exclusive_run, read_json, seed_for, write_json

PROJECT = Path(__file__).resolve().parents[2]


def initialize(config, directory, summary, client):
    from importlib.metadata import version
    source_files = sorted((PROJECT / "src").rglob("*.py")) + sorted((PROJECT / "prompts").glob("*.txt"))
    source_hash = digest({str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files})
    stamp = {"config": config, "manifests": summary["manifest_hashes"], "source_hash": source_hash,
             "model": client.metadata(),
             "software": {"python": platform.python_version(),
                          **{name: version(name) for name in ("alfworld", "textworld", "fast-downward-textworld", "tokenizers", "jsonschema")}},
             "tokenizer_sha256": hashlib.sha256((PROJECT / config["tokenizer"]).read_bytes()).hexdigest()}
    path = directory / "run_metadata.json"
    if path.exists() and read_json(path) != stamp:
        raise ValueError("代码、配置、模型或数据清单与已有运行不同。请使用新的 --run-dir，不能混合实验。")
    write_json(path, stamp)
    for name in ("plan.json", "development.json", "summary.json"):
        write_json(directory / "manifests" / name, read_json(PROJECT / "data/manifests" / name))


def report(directory):
    rounds = [read_json(path) for path in sorted((directory / "rounds").glob("*/result.json"))]
    test = read_json(directory / "test_results.json") if (directory / "test_results.json").exists() else None
    value = {"completed_rounds": len(rounds), "accepted_rounds": sum(r["status"] == "accepted" for r in rounds),
             "rounds": [{"round": r["round"], "status": r["status"], "memory_version": r["memory_after"]["version"]}
                        for r in rounds], "heldout_test": test}
    write_json(directory / "report.json", value)
    lines = ["# ALFWorld 实验结果", "", f"完成进化轮数：{len(rounds)}；接收经验轮数：{value['accepted_rounds']}。", "",
             "| 轮次 | 状态 | 经验版本 |", "| --- | --- | --- |"]
    lines += [f"| {r['round']} | {r['status']} | {r['memory_version']} |" for r in value["rounds"]]
    if test:
        lines += ["", f"最终测试题数：{test['tasks']}。", "", "| 经验库 | 成功数 | 成功率 | 平均步数 |", "| --- | --- | --- | --- |"]
        for name, metric in test["metrics"].items():
            lines.append(f"| {name} | {metric['successes']} | {metric['success_rate']:.1%} | {metric['mean_steps']:.2f} |")
    else:
        lines += ["", "最终测试尚未完成；不能据此报告测试成功率。"]
    lines += ["", "原题通过后，扩展轨迹仅用于修正经验；修正后直接追加，未做复验。最终效果以冻结测试结果为准。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return value


def check(config, summary, client):
    from importlib.metadata import version
    plan = read_json(PROJECT / "data/manifests/plan.json")
    task = plan[0]["original"]
    game = PROJECT / config["data_root"] / task["game_path"]
    env = AlfworldEnvironment(game, seed_for(config["seed"], "environment", task["id"]))
    try:
        first, won = env.reset()
        env.step("look")
        second, second_won = env.reset()
    finally:
        env.close()
    if first != second or won != second_won:
        raise RuntimeError("ALFWorld reset did not reproduce the initial state")
    result = {"versions": {name: version(name) for name in ("alfworld", "textworld", "fast-downward-textworld")},
              "pools": summary["actual"], "rounds": summary["rounds"], "reset_reproducible": True,
              "model": client.metadata()}
    write_json(PROJECT / "logs/installation_check.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="ALFWorld 轨迹对比与经验进化实验")
    parser.add_argument("command", choices=["prepare", "check", "smoke", "pilot", "evolve", "test", "run", "status", "report"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = yaml.safe_load(config_path.read_text())
    default_directory = {"smoke": "runs/smoke", "pilot": "runs/pilot"}.get(args.command, "runs/main")
    directory = Path(args.run_dir or default_directory)
    if not directory.is_absolute():
        directory = PROJECT / directory
    os.environ["ALFWORLD_DATA"] = str(PROJECT / "data/alfworld")
    if args.command == "status":
        if (directory / "status.json").exists():
            print(json.dumps(read_json(directory / "status.json"), ensure_ascii=False, indent=2))
            log = directory / "events.jsonl"
            if log.exists():
                with log.open() as stream:
                    from collections import deque
                    for line in deque(stream, maxlen=3):
                        print(line.rstrip())
        else:
            print("该目录尚未启动正式实验。")
        return
    if args.command == "report":
        print(json.dumps(report(directory), ensure_ascii=False, indent=2))
        return
    summary = prepare(config, PROJECT)
    if args.command == "prepare":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        config = deepcopy(config)
        config["episode"]["max_steps"] = 2
    with exclusive_run(directory):
        client = OllamaClient(config, PROJECT, directory)
        if args.command == "check":
            print(json.dumps(check(config, summary, client), ensure_ascii=False, indent=2))
            return
        initialize(config, directory, summary, client)
        runner = EpisodeRunner(config, PROJECT, directory, client)
        cognition = Cognition(PROJECT, client, config["seed"])
        plan = read_json(directory / "manifests/plan.json")
        if args.command == "pilot":
            trace = runner.run(plan[0]["original"], Memory(), seed_for(config["seed"], "development-pilot"),
                               "development-pilot", "diagnostic-only")
            result = {"diagnostic_only": True, "not_a_benchmark_result": True,
                      "task_id": trace["task"]["id"], "success": trace["success"],
                      "steps": len(trace["steps"]), "termination": trace["termination"]}
            write_json(directory / "pilot_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "smoke":
            trace = runner.run(plan[0]["original"], Memory(), seed_for(config["seed"], "smoke"),
                               "smoke-actor", "diagnostic-only")
            response = client.call([{"role": "user", "content": 'Return {"status":"ok"}.'}],
                                   seed_for(config["seed"], "smoke-json"), "smoke-structured-output",
                                   object_schema({"status": {"enum": ["ok"]}}))
            result = {"diagnostic_only": True, "not_a_benchmark_result": True,
                      "environment_steps": len(trace["steps"]), "structured_output": response,
                      "actor_and_reflection_contexts": [config["model"]["actor_context"], config["model"]["reflection_context"]]}
            write_json(directory / "smoke_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command in {"evolve", "run"}:
            Evolution(config, directory, runner, cognition).evolve(plan)
        if args.command in {"test", "run"}:
            # The test manifest is read only after evolution and freezing; never supplied to cognition.
            frozen_path = directory / "memory_final.json"
            if not frozen_path.exists() or not read_json(frozen_path).get("frozen"):
                raise ValueError("必须先完成进化并冻结经验库，才能读取和运行最终测试清单。")
            tasks = read_json(PROJECT / "data/manifests/test.json")["test"]
            write_json(directory / "manifests/test.json", {"test": tasks})
            evaluate(config, directory, tasks, runner)
        report(directory)


def entrypoint():
    def stop(_signal, _frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        main()
    except KeyboardInterrupt:
        print("\n实验已中断。已完成的模型调用和轨迹保留；再次运行同一命令会续跑。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n实验停止：{type(exc).__name__}: {exc}\n环境或 API 故障不会记为任务失败。修复后可续跑。", file=sys.stderr)
        raise


if __name__ == "__main__":
    entrypoint()
