import argparse
import os
from pathlib import Path

import yaml

from .client import Ollama
from .experiment import Experiment, report
from .io import dumps, read_json, write_json
from .memory import Memory
from .roles import Roles

PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description="V2: Planner / Actor / Forecaster / Critic")
    parser.add_argument("command", choices=["check", "smoke", "train", "evaluate", "run", "status", "report"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load((PROJECT / args.config).read_text())
    directory = PROJECT / (args.run_dir or ("runs/smoke" if args.command == "smoke" else "runs/main"))
    if args.command == "status":
        print(dumps(read_json(directory / "status.json")))
        return
    if args.command == "report":
        print(dumps(report(directory)))
        return
    if args.command == "smoke":
        args.limit, args.max_steps = 1, 2
    if args.max_steps is not None:
        config["episode"]["max_steps"] = args.max_steps
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "config.yaml").exists():
        (directory / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    os.environ["ALFWORLD_DATA"] = str((PROJECT / config["data_root"]).parent.resolve())
    tasks = read_json(PROJECT / config["tasks"])
    client = Ollama(config, PROJECT, directory)
    roles = Roles(client, PROJECT / "prompts")
    experiment = Experiment(config, PROJECT, directory, roles)
    if args.command == "check":
        metadata = client.metadata()
        embeddings = client.embed(["一条可见状态", "一个待解决问题"])
        env = experiment.new_environment(tasks["train"][0])
        try:
            initial, _ = env.reset()
        finally:
            env.close()
        value = {"metadata": metadata, "embedding_shape": list(embeddings.shape),
                 "environment_reset": bool(initial), "tasks": {k: len(v) for k, v in tasks.items()}}
        write_json(directory / "check.json", value)
        print(dumps(value))
        return
    if not (directory / "metadata.json").exists():
        write_json(directory / "metadata.json", client.metadata())
    if args.command in {"train", "run", "smoke"}:
        train_tasks = tasks["train"][:args.limit]
        write_json(directory / "training_tasks.json", train_tasks)
        memory = experiment.train(train_tasks)
    else:
        memory = Memory(read_json(directory / "memory_final.json"))
    if args.command in {"evaluate", "run"}:
        test_tasks = tasks["test"][:args.limit]
        write_json(directory / "evaluation_tasks.json", test_tasks)
        experiment.evaluate(test_tasks, memory)
    if args.command == "smoke":
        write_json(directory / "diagnostic.json", {"diagnostic_only": True, "tasks": 1,
                                                   "plans": 3, "max_steps": 2})
    print(dumps(report(directory)))


if __name__ == "__main__":
    main()
