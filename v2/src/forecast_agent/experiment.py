from pathlib import Path

from .environment import Alfworld, task_goal
from .io import dumps, event, read_json, seed_for, write_json
from .memory import Memory


class Experiment:
    def __init__(self, config, project, directory, roles, environment=Alfworld):
        self.config, self.project, self.directory = config, Path(project), Path(directory)
        self.roles, self.client, self.environment = roles, roles.client, environment
        self.seed = config["seed"]

    def new_environment(self, task):
        return self.environment(self.project / self.config["data_root"] / task["game_path"],
                                seed_for(self.seed, "environment", task["id"]))

    def injection(self, memory, target, context, seed, purpose):
        return memory.select(target, context, self.roles, self.config["memory"]["full_threshold"],
                             seed, purpose, self.directory)

    def actor_messages(self, initial, plan, experiences, steps):
        history = []
        for step in reversed(steps):
            candidate = [step, *history]
            if self.client.tokens(dumps(candidate)) > self.config["episode"]["history_tokens"]:
                break
            history = candidate
        return [{"role": "system", "content": self.roles.prompts["actor"]},
                {"role": "user", "content": dumps({"task_goal": task_goal(initial),
                 "initial_observation": initial, "plan": plan,
                 "experiences": experiences, "omitted_history_steps": len(steps) - len(history),
                 "recent_interaction": history})}]

    def episode(self, task, plan, memory, plan_index, label, sampling_tag):
        identifier = f"{label}-plan{plan_index + 1}"
        path = self.directory / "episodes" / (identifier + ".json")
        if path.exists():
            return read_json(path)
        env = self.new_environment(task)
        try:
            initial, won = env.reset()
            context = {"initial_observation": initial, "plan": plan}
            injection = self.injection(memory, "actor", context,
                                      seed_for(self.seed, sampling_tag, task["id"], plan_index, "forecast-actor"),
                                      identifier + "/inject")
            trace = {"id": identifier, "task_id": task["id"], "plan": plan,
                     "initial_observation": initial, "memory_version": memory.data["version"],
                     "injection": injection, "steps": [], "won": won, "termination": None}
            progress = path.with_suffix(".progress.json")
            event(self.directory, "episode_start", episode_id=identifier, memory_version=memory.data["version"])
            print(f"  {identifier}: start", flush=True)
            for index in range(self.config["episode"]["max_steps"]):
                if won:
                    break
                messages = self.actor_messages(initial, plan, injection["experiences"], trace["steps"])
                action = self.roles.act(messages,
                                        seed_for(self.seed, sampling_tag, task["id"], plan_index, "action", index),
                                        identifier + f"/action-{index + 1}")
                observation, score, done, won = env.step(action)
                trace["steps"].append({"action": action, "observation": observation, "score": score,
                                       "done": done, "won": won})
                trace["won"] = won
                write_json(progress, trace)
                event(self.directory, "step", episode_id=identifier, step=index + 1, action=action, won=won)
                print(f"    step {index + 1}: {action}", flush=True)
                if done:
                    trace["termination"] = "won" if won else "environment_done"
                    break
            trace["termination"] = trace["termination"] or ("won" if won else "step_limit")
            write_json(path, trace)
            progress.unlink(missing_ok=True)
            event(self.directory, "episode_end", episode_id=identifier, won=won, steps=len(trace["steps"]))
            return trace
        finally:
            env.close()

    def solve(self, task, memory, label, sampling_tag):
        env = self.new_environment(task)
        try:
            initial, _ = env.reset()
        finally:
            env.close()
        injection = self.injection(memory, "planner", {"initial_observation": initial},
                                  seed_for(self.seed, sampling_tag, task["id"], "forecast-planner"),
                                  label + "/inject-planner")
        plans = self.roles.plan(initial, injection["experiences"],
                                seed_for(self.seed, sampling_tag, task["id"], "planner"), label + "/planner")
        episodes = [self.episode(task, plan, memory, i, label, sampling_tag) for i, plan in enumerate(plans)]
        return {"task_id": task["id"], "initial_observation": initial,
                "planning_injection": injection, "plans": plans, "episodes": episodes}

    def train(self, tasks):
        memory = Memory()
        for index, task in enumerate(tasks, 1):
            path = self.directory / "rounds" / f"{index:03d}.json"
            if path.exists():
                memory = Memory(read_json(path)["memory_after"])
                continue
            write_json(self.directory / "status.json", {"phase": "train", "round": index, "total": len(tasks)})
            result = self.solve(task, memory, f"train-{index:03d}", "train")
            critique = self.roles.critique(result["initial_observation"], result["plans"], result["episodes"],
                                          seed_for(self.seed, "critic", index), f"round-{index:03d}/critic")
            after = memory.updated(critique, self.client, self.config["memory"]["similarity_threshold"])
            # The round file is the commit: the three episodes only read the old snapshot.
            write_json(path, {**result, "critique": critique, "memory_after": after.data})
            memory = after
            write_json(self.directory / "memory.json", memory.data)
            event(self.directory, "memory_commit", round=index, version=memory.data["version"],
                  experiences=len(memory.data["experiences"]), clusters=len(memory.data["clusters"]))
            print(f"Round {index}/{len(tasks)}: {len(memory.data['experiences'])} experiences, "
                  f"{len(memory.data['clusters'])} clusters", flush=True)
        write_json(self.directory / "memory.json", memory.data)
        write_json(self.directory / "memory_final.json", memory.data)
        write_json(self.directory / "status.json", {"phase": "trained", "completed": len(tasks), "total": len(tasks)})
        return memory

    def evaluate(self, tasks, memory):
        outcomes = {}
        for name, snapshot in (("initial", Memory()), ("final", memory)):
            solved = []
            snapshot_name = f"{name}-v{snapshot.data['version']}"
            for index, task in enumerate(tasks, 1):
                path = self.directory / "evaluation" / f"{snapshot_name}-{index:03d}.json"
                write_json(self.directory / "status.json", {"phase": "evaluate", "memory": name,
                                                            "completed": index - 1, "total": len(tasks)})
                if path.exists():
                    result = read_json(path)
                else:
                    result = self.solve(task, snapshot, f"test-{snapshot_name}-{index:03d}", "evaluate")
                    write_json(path, result)
                solved.append(result)
            outcomes[name] = solved
        result = {"outcomes": outcomes, "metrics": {name: metrics(rows) for name, rows in outcomes.items()}}
        write_json(self.directory / "evaluation.json", result)
        write_json(self.directory / "status.json", {"phase": "complete", "metrics": result["metrics"]})
        return result


def metrics(rows):
    episodes = [e for row in rows for e in row["episodes"]]
    successful = [e for e in episodes if e["won"]]
    return {"tasks": len(rows), "trajectories": len(episodes), "successes": len(successful),
            "success_rate": len(successful) / len(episodes),
            "task_any_success_rate": sum(any(e["won"] for e in r["episodes"]) for r in rows) / len(rows),
            "mean_steps": sum(len(e["steps"]) for e in episodes) / len(episodes),
            "successful_mean_steps": (sum(len(e["steps"]) for e in successful) / len(successful)
                                      if successful else None)}


def report(directory):
    directory = Path(directory)
    rounds = [read_json(p) for p in sorted((directory / "rounds").glob("*.json"))]
    calls = [read_json(p) for p in (directory / "model_calls").glob("*.json")]
    result = {"completed_rounds": len(rounds),
              "training": metrics(rounds) if rounds else None,
              "model_calls": len(calls),
              "prompt_tokens": sum(c["response"].get("prompt_eval_count", 0) for c in calls),
              "output_tokens": sum(c["response"].get("eval_count", 0) for c in calls)}
    if (directory / "evaluation.json").exists():
        result["evaluation"] = read_json(directory / "evaluation.json")["metrics"]
    lines = ["# V2 实验结果", "", f"已完成训练题：{len(rounds)}。", "",
             "三方案分别重置执行；任务成功以环境 won 为准。训练结束后冻结经验，评测不更新经验。", ""]
    if rounds:
        memory = rounds[-1]["memory_after"]
        lines += [f"经验条目：{len(memory['experiences'])}；经验簇：{len(memory['clusters'])}。", ""]
    if "evaluation" in result:
        lines += ["| 经验库 | 执行成功数 | 执行成功率 | 三方案至少一次成功的题目比例 | 平均步数 |", "| --- | --- | --- | --- | --- |"]
        for name, m in result["evaluation"].items():
            lines.append(f"| {name} | {m['successes']}/{m['trajectories']} | {m['success_rate']:.1%} | "
                         f"{m['task_any_success_rate']:.1%} | {m['mean_steps']:.2f} |")
        lines += ["", "执行成功率与三方案覆盖率分别报告；不把三次中任意一次成功当作单次执行成功率。", ""]
    lines += [f"累计唯一模型调用：{len(calls)}；输入 token：{result['prompt_tokens']}；输出 token：{result['output_tokens']}。", ""]
    write_json(directory / "report.json", result)
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
