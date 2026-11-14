from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from forecast_agent.experiment import Experiment, metrics
from forecast_agent.io import dumps, read_json
from forecast_agent.memory import Memory, vectors


def content(context="尚未确认前序结果", problem="缺少后续需要的信息"):
    return {"context": context, "problem": problem, "explanation": "暂时解释",
            "methods": ["确认信息后继续"], "advice": ["根据反馈决定后续动作"], "exceptions": []}


class FakeClient:
    def tokens(self, text):
        return len(text) // 4

    def embed(self, texts):
        return np.asarray([[1., 0.] if text in ("A", "尚未确认前序结果", "缺少后续需要的信息")
                           else [0., 1.] for text in texts])


class FakeRoles:
    def __init__(self):
        self.client = FakeClient()
        self.prompts = {"actor": "Return an action."}
        self.forecasts = []
        self.critiques = 0
        self.consolidations = 0
        self.actions = []
        self.choose = None

    def forecast(self, phase, context, judgments, catalogue, seed, purpose):
        self.forecasts.append((phase, deepcopy(context), deepcopy(judgments), deepcopy(catalogue)))
        return {"possible_problems": ["候选问题"],
                "selected_project_ids": self.choose if self.choose is not None else [c["id"] for c in catalogue]}

    def plan(self, initial, experiences, seed, purpose):
        return [{"name": str(i), "strategy": f"strategy {i}", "steps": ["探索", "完成任务"]} for i in range(3)]

    def act(self, messages, seed, purpose):
        self.actions.append((deepcopy(messages), purpose))
        return "look"

    def critique(self, initial, plans, traces, seed, purpose):
        self.critiques += 1
        return {"judge_experiences": [{**content(), "phase": phase} for phase in ("planning", "execution")],
                "planner_experiences": [content()], "actor_experiences": [content()]}

    def consolidate(self, clusters, seed, purpose):
        self.consolidations += 1
        return {c["id"]: {"state_name": "训练中归纳的状态", "problem_name": "训练中归纳的问题", "scope": "适用条件",
                            "experiences": [{k: v for k, v in e.items() if k != "id"} for e in c["experiences"]]}
                for c in clusters}


class FakeEnvironment:
    instances = []

    def __init__(self, path, seed):
        self.actions = []
        self.closed = False
        self.instances.append(self)

    def reset(self):
        self.actions = []
        return "Visible initial state and task.", False

    def step(self, action):
        self.actions.append(action)
        won = len(self.actions) == 2
        return f"Feedback {len(self.actions)}", float(won), won, won

    def close(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.roles = FakeRoles()

    def memory_with(self, count, target="actor", phase="execution"):
        memory = Memory()
        entries = [memory.add(target, phase, content()) for _ in range(count)]
        for e in entries:
            e["cluster_id"] = "C1"
        memory.data["clusters"] = [{"id": "C1", "type": target, "phase": phase,
            "state_name": "状态", "problem_name": "问题", "scope": "条件",
            "experience_ids": [e["id"] for e in entries], "centroid": vectors(self.roles.client, entries).mean(axis=0).tolist()}]
        memory.data["next_cluster"] = 2
        return memory

    def test_below_threshold_injects_all_without_forecaster(self):
        memory = self.memory_with(19)
        selected = memory.select("actor", {"plan": "p"}, self.roles, 20, 1, "full", self.directory)
        self.assertEqual(selected["mode"], "full")
        self.assertEqual(len(selected["experiences"]), 19)
        self.assertEqual(self.roles.forecasts, [])

    def test_at_threshold_reads_selected_project_in_full(self):
        memory = self.memory_with(20)
        selected = memory.select("actor", {}, self.roles, 20, 1, "catalogue", self.directory)
        self.assertEqual(selected["mode"], "catalogue")
        self.assertEqual(len(selected["experiences"]), 20)
        self.assertEqual(self.roles.forecasts[0][0], "execution")

    def test_unselected_project_is_not_injected(self):
        memory = self.memory_with(9)
        extra = memory.add("actor", "execution", content("B", "B"))
        extra["cluster_id"] = "C2"
        memory.data["clusters"].append({**memory.data["clusters"][0], "id": "C2", "experience_ids": [extra["id"]]})
        self.roles.choose = ["C1"]
        selected = memory.select("actor", {}, self.roles, 1, 1, "select", self.directory)
        self.assertEqual(len(selected["experiences"]), 9)
        self.assertNotIn(extra["id"], [e["id"] for e in selected["experiences"]])

    def test_forecaster_can_select_no_projects(self):
        memory = self.memory_with(2)
        self.roles.choose = []
        selected = memory.select("actor", {}, self.roles, 1, 1, "none", self.directory)
        self.assertEqual(selected["experiences"], [])

    def test_judgment_phase_matches_receiving_module(self):
        memory = self.memory_with(1, "planner", "planning")
        memory.add("judge", "planning", content("planning-only", "risk"))
        memory.add("judge", "execution", content("execution-only", "risk"))
        memory.select("planner", {}, self.roles, 1, 1, "planner", self.directory)
        phase, _, judgments, _ = self.roles.forecasts[0]
        self.assertEqual(phase, "planning")
        self.assertEqual([e["context"] for e in judgments], ["planning-only"])

    def test_embedding_uses_both_context_and_problem(self):
        encoded = vectors(self.roles.client, [content("A", "A"), content("A", "B"), content("B", "A")])
        self.assertAlmostEqual(float(encoded[0] @ encoded[1]), 0.5)
        self.assertAlmostEqual(float(encoded[0] @ encoded[2]), 0.5)

    def test_four_collections_remain_separate_after_clustering(self):
        before = Memory()
        critique = self.roles.critique("", [], [], 0, "")
        after = before.updated(critique, self.roles.client, self.roles, .8, 42, 1)
        self.assertEqual(before.data["version"], 0)
        self.assertEqual(before.data["experiences"], [])
        self.assertEqual(len(after.data["clusters"]), 4)
        self.assertEqual(self.roles.consolidations, 0)
        self.assertEqual({(c["type"], c["phase"]) for c in after.data["clusters"]},
                         {("judge", "planning"), ("judge", "execution"),
                          ("planner", "planning"), ("actor", "execution")})
        for e in after.data["experiences"]:
            self.assertNotIn("evidence", e)
            self.assertNotIn("sources", e)

    def test_new_method_can_join_existing_cluster(self):
        memory = self.memory_with(1)
        fresh = content()
        fresh["methods"] = ["另一种方法"]
        after = memory.updated({"judge_experiences": [], "planner_experiences": [], "actor_experiences": [fresh]},
                               self.roles.client, self.roles, .8, 42, 1)
        self.assertEqual(len(after.data["clusters"]), 1)
        self.assertEqual({m for e in after.data["experiences"] for m in e["methods"]},
                         {"确认信息后继续", "另一种方法"})

    def test_unrelated_experience_creates_a_cluster(self):
        memory = self.memory_with(1)
        after = memory.updated({"judge_experiences": [], "planner_experiences": [],
                                "actor_experiences": [content("B", "B")]},
                               self.roles.client, self.roles, .8, 42, 1)
        self.assertEqual(len(after.data["clusters"]), 2)

    def experiment(self):
        config = {"seed": 42, "data_root": "data", "episode": {"max_steps": 2, "history_tokens": 2000},
                  "memory": {"full_threshold": 1, "similarity_threshold": .8}}
        return Experiment(config, self.directory, self.directory, self.roles, FakeEnvironment)

    def test_training_injects_once_per_plan_and_commits_once_per_task(self):
        experiment = self.experiment()
        tasks = [{"id": f"task{i}", "game_path": f"task{i}.json"} for i in range(2)]
        memory = experiment.train(tasks)
        self.assertEqual(memory.data["version"], 2)
        self.assertEqual(self.roles.critiques, 2)
        self.assertEqual(self.roles.consolidations, 1)
        self.assertEqual(len(self.roles.forecasts), 4)  # Second task: Planner once, Actor three times.
        self.assertEqual([f[0] for f in self.roles.forecasts], ["planning", "execution", "execution", "execution"])
        actor_contexts = [f[1] for f in self.roles.forecasts if f[0] == "execution"]
        self.assertTrue(all(set(c) == {"initial_observation", "plan"} for c in actor_contexts))
        events = [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e["event"] == "memory_commit" for e in events), 2)
        self.assertEqual(sum(e["event"] == "injection" and e["target"] == "actor" for e in events), 6)
        episodes = [read_json(p) for p in (self.directory / "episodes").glob("*.json")]
        self.assertEqual(len(episodes), 6)
        self.assertTrue(all(len(e["steps"]) == 2 for e in episodes))
        self.assertEqual([e["memory_version"] for e in episodes].count(0), 3)
        self.assertEqual([e["memory_version"] for e in episodes].count(1), 3)

    def test_resume_does_not_repeat_completed_tasks_or_memory_writes(self):
        experiment = self.experiment()
        tasks = [{"id": "task", "game_path": "task.json"}]
        memory = experiment.train(tasks)
        count = len(self.roles.actions)
        resumed = experiment.train(tasks)
        self.assertEqual(memory.data, resumed.data)
        self.assertEqual(len(self.roles.actions), count)
        self.assertEqual(self.roles.critiques, 1)

    def test_evaluation_keeps_memory_frozen_and_uses_equal_three_plan_budgets(self):
        experiment = self.experiment()
        tasks = [{"id": "task", "game_path": "task.json"}]
        memory = experiment.train(tasks)
        before = dumps(memory.data)
        critiques = self.roles.critiques
        result = experiment.evaluate(tasks, memory)
        self.assertEqual(dumps(memory.data), before)
        self.assertEqual(self.roles.critiques, critiques)
        self.assertEqual(result["metrics"]["initial"]["trajectories"], 3)
        self.assertEqual(result["metrics"]["final"]["trajectories"], 3)

    def test_three_plan_coverage_is_separate_from_execution_success(self):
        rows = [{"episodes": [{"won": True, "steps": [1]}, {"won": False, "steps": [1, 2]},
                               {"won": False, "steps": [1, 2]}]}]
        result = metrics(rows)
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        self.assertEqual(result["task_any_success_rate"], 1)

    def test_new_frozen_version_does_not_reuse_previous_final_evaluation(self):
        experiment = self.experiment()
        tasks = [{"id": "task", "game_path": "task.json"}]
        memory = experiment.train(tasks)
        experiment.evaluate(tasks, memory)
        action_count = len(self.roles.actions)
        memory.data["version"] += 1
        result = experiment.evaluate(tasks, memory)
        self.assertEqual(len(self.roles.actions) - action_count, 6)
        self.assertTrue(all(e["memory_version"] == 2 for e in result["outcomes"]["final"][0]["episodes"]))


if __name__ == "__main__":
    unittest.main()
