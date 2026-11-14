from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from forecast_agent.experiment import Experiment, metrics
from forecast_agent.io import dumps, read_json
from forecast_agent.memory import Memory, vectors
from forecast_agent.roles import Roles


def content(state="尚未确认前序结果，缺少后续需要的信息"):
    return {"state": state, "explanation": "暂时解释",
            "advice": ["根据反馈决定后续动作"], "exceptions": []}


class FakeClient:
    def __init__(self):
        self.embedded = []

    def tokens(self, text):
        return len(text) // 4

    def embed(self, texts):
        self.embedded.append(list(texts))
        known = {"A": [1., 0.], "尚未确认前序结果，缺少后续需要的信息": [1., 0.],
                 "near": [4., 3.], "middle": [3., 1.]}
        return np.asarray([known.get(text, [0., 1.]) for text in texts])


class FakeRoles:
    def __init__(self):
        self.client = FakeClient()
        self.prompts = {"actor": "Return an action."}
        self.forecasts = []
        self.critiques = 0
        self.actions = []
        self.choose = None

    def forecast(self, phase, context, judgments, catalogue, seed, purpose):
        self.forecasts.append((phase, deepcopy(context), deepcopy(judgments), deepcopy(catalogue)))
        return {"possible_problems": ["候选问题"],
                "selected_state_ids": self.choose if self.choose is not None else [c["state_id"] for c in catalogue]}

    def plan(self, initial, experiences, seed, purpose):
        return [{"name": str(i), "strategy": f"strategy {i}", "steps": ["探索", "完成任务"]} for i in range(3)]

    def act(self, messages, seed, purpose):
        self.actions.append((deepcopy(messages), purpose))
        return "look"

    def critique(self, initial, plans, traces, seed, purpose):
        self.critiques += 1
        return {"judge_experiences": [{**content(), "phase": phase} for phase in ("planning", "execution")],
                "planner_experiences": [content()], "actor_experiences": [content()]}

class FakeEnvironment:
    instances = []

    def __init__(self, path, seed):
        self.actions = []
        self.closed = False
        self.instances.append(self)

    def reset(self):
        self.actions = []
        return "Visible initial state.\nYour task is to: find the target object.", False

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
            e["state_id"] = "S00001"
        memory.data["clusters"] = [{"state_id": "S00001", "type": target, "phase": phase,
            "state": entries[0]["state"],
            "experience_ids": [e["id"] for e in entries], "centroid": vectors(self.roles.client, entries).mean(axis=0).tolist()}]
        memory.data["next_state"] = 2
        return memory

    def test_below_threshold_injects_all_without_forecaster(self):
        memory = self.memory_with(7)
        selected = memory.select("actor", {"plan": "p"}, self.roles, 8, 1, "full", self.directory)
        self.assertEqual(selected["mode"], "full")
        self.assertEqual(len(selected["experiences"]), 7)
        self.assertEqual(self.roles.forecasts, [])

    def test_at_threshold_reads_selected_state_in_full(self):
        memory = self.memory_with(8)
        selected = memory.select("actor", {}, self.roles, 8, 1, "catalogue", self.directory)
        self.assertEqual(selected["mode"], "catalogue")
        self.assertEqual(len(selected["experiences"]), 8)
        self.assertEqual(selected["state_ids"], ["S00001"])
        self.assertEqual(self.roles.forecasts[0][0], "execution")

    def test_selected_state_injects_all_members_and_excludes_other_states(self):
        memory = self.memory_with(13)
        extra = memory.add("actor", "execution", content("B"))
        extra["state_id"] = "S00002"
        memory.data["clusters"].append({**memory.data["clusters"][0], "state_id": "S00002",
                                        "state": "B", "experience_ids": [extra["id"]]})
        self.roles.choose = ["S00001"]
        selected = memory.select("actor", {}, self.roles, 8, 1, "select", self.directory)
        self.assertEqual(len(selected["experiences"]), 13)
        self.assertNotIn(extra["id"], [e["id"] for e in selected["experiences"]])

    def test_forecaster_receives_only_target_state_index(self):
        memory = self.memory_with(8)
        memory.select("actor", {"plan": "p"}, self.roles, 8, 1, "catalogue", self.directory)
        _, context, _, catalogue = self.roles.forecasts[0]
        self.assertEqual(context, {"plan": "p"})
        self.assertEqual(catalogue, [{"state_id": "S00001", "state": content()["state"]}])

    def test_forecaster_can_select_no_states(self):
        memory = self.memory_with(8)
        self.roles.choose = []
        selected = memory.select("actor", {}, self.roles, 8, 1, "none", self.directory)
        self.assertEqual(selected["experiences"], [])

    def test_judgment_phase_matches_receiving_module(self):
        memory = self.memory_with(8, "planner", "planning")
        memory.add("judge", "planning", content("planning-only"))
        memory.add("judge", "execution", content("execution-only"))
        memory.select("planner", {}, self.roles, 8, 1, "planner", self.directory)
        phase, _, judgments, _ = self.roles.forecasts[0]
        self.assertEqual(phase, "planning")
        self.assertEqual([e["state"] for e in judgments], ["planning-only"])

    def test_embedding_uses_only_state_and_ignores_experience_body(self):
        changed = {**content("A"), "explanation": "完全不同的解释", "advice": ["另一种做法"],
                   "exceptions": ["另一种例外"]}
        encoded = vectors(self.roles.client, [content("A"), changed, content("B")])
        self.assertEqual(self.roles.client.embedded[-1], ["A", "A", "B"])
        np.testing.assert_allclose(encoded[0], encoded[1])
        self.assertAlmostEqual(float(encoded[0] @ encoded[2]), 0.)
        np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1.)

    def test_four_collections_remain_separate_after_clustering(self):
        before = Memory()
        critique = self.roles.critique("", [], [], 0, "")
        after = before.updated(critique, self.roles.client, .8)
        self.assertEqual(before.data["version"], 0)
        self.assertEqual(before.data["experiences"], [])
        self.assertEqual(len(after.data["clusters"]), 4)
        self.assertEqual({(c["type"], c["phase"]) for c in after.data["clusters"]},
                         {("judge", "planning"), ("judge", "execution"),
                          ("planner", "planning"), ("actor", "execution")})
        for e in after.data["experiences"]:
            self.assertNotIn("evidence", e)
            self.assertNotIn("sources", e)

    def test_different_advice_joins_same_state_without_rewriting_originals(self):
        memory = self.memory_with(1)
        original = deepcopy(memory.data["experiences"][0])
        fresh = content()
        fresh["explanation"] = "另一个解释"
        fresh["advice"] = ["另一种做法"]
        after = memory.updated({"judge_experiences": [], "planner_experiences": [], "actor_experiences": [fresh]},
                               self.roles.client, .8)
        self.assertEqual(len(after.data["clusters"]), 1)
        self.assertEqual(after.data["experiences"][0], original)
        self.assertEqual(memory.data["experiences"], [original])
        self.assertEqual({a for e in after.data["experiences"] for a in e["advice"]},
                         {"根据反馈决定后续动作", "另一种做法"})
        self.assertEqual({e["explanation"] for e in after.data["experiences"]}, {"暂时解释", "另一个解释"})
        self.assertEqual(after.data["clusters"][0]["experience_ids"], [e["id"] for e in after.data["experiences"]])

    def test_state_index_uses_member_nearest_centroid(self):
        critique = {"judge_experiences": [], "planner_experiences": [],
                    "actor_experiences": [content("A"), content("near"), content("middle")]}
        after = Memory().updated(critique, self.roles.client, .75)
        self.assertEqual(len(after.data["clusters"]), 1)
        cluster = after.data["clusters"][0]
        self.assertEqual(cluster["state"], "middle")
        self.assertEqual([e["state"] for e in after.data["experiences"]], ["A", "near", "middle"])
        center = vectors(self.roles.client, after.data["experiences"]).mean(axis=0)
        np.testing.assert_allclose(cluster["centroid"], center / np.linalg.norm(center))

    def test_unrelated_experience_creates_a_cluster(self):
        memory = self.memory_with(1)
        after = memory.updated({"judge_experiences": [], "planner_experiences": [],
                                "actor_experiences": [content("B")]},
                               self.roles.client, .8)
        self.assertEqual(len(after.data["clusters"]), 2)

    def experiment(self):
        config = {"seed": 42, "data_root": "data", "episode": {"max_steps": 2, "history_tokens": 2000},
                  "memory": {"full_threshold": 1, "similarity_threshold": .8}}
        return Experiment(config, self.directory, self.directory, self.roles, FakeEnvironment)

    def test_planner_preserves_raw_goal_when_experience_mentions_another_task(self):
        goal = "put a clean Plate in/on the DiningTable."
        initial = "You are in the middle of a room.\nYour task is to: " + goal
        experiences = [{**content(), "advice": ["Your task is to: heat a mug and put it in/on the fridge."]}]
        client = Mock()
        client.chat.return_value = {"plans": []}
        roles = Roles(client, Path(__file__).resolve().parents[1] / "prompts")

        roles.plan(initial, experiences, 42, "planner-goal")

        payload = json.loads(client.chat.call_args.args[1][1]["content"])
        self.assertEqual(payload["task_goal"], goal)
        self.assertEqual(payload["initial_observation"], initial)
        self.assertEqual(payload["experiences"], experiences)

    def test_critic_preserves_raw_goal_when_plan_mentions_another_task(self):
        goal = "put a clean Plate in/on the DiningTable."
        initial = "You are in the middle of a room.\nYour task is to: " + goal
        plans = [{"name": "other goal", "strategy": "heat a mug and put it in/on the fridge.", "steps": []}]
        traces = [{"won": False, "termination": "step_limit", "steps": []}]
        client = Mock()
        client.chat.return_value = {"judge_experiences": [], "planner_experiences": [], "actor_experiences": []}
        roles = Roles(client, Path(__file__).resolve().parents[1] / "prompts")

        roles.critique(initial, plans, traces, 42, "critic-goal")

        payload = json.loads(client.chat.call_args.args[1][1]["content"])
        self.assertEqual(payload["task_goal"], goal)
        self.assertEqual(payload["initial_observation"], initial)
        self.assertEqual(payload["plans"], plans)

    def test_actor_keeps_raw_goal_when_recent_history_is_trimmed(self):
        experiment = self.experiment()
        experiment.config["episode"]["history_tokens"] = 50
        goal = "put a clean Plate in/on the DiningTable."
        initial = "You are in the middle of a room.\nYour task is to: " + goal
        plan = {"name": "other goal", "strategy": "heat a mug and put it in/on the fridge.", "steps": []}
        experiences = [{**content(), "advice": ["Your task is to: heat a mug and put it in/on the fridge."]}]
        steps = [{"action": "look", "observation": "old feedback " * 100, "won": False},
                 {"action": "look", "observation": "latest feedback", "won": False}]

        early = json.loads(experiment.actor_messages(initial, plan, experiences, [])[1]["content"])
        later = json.loads(experiment.actor_messages(initial, plan, experiences, steps)[1]["content"])

        self.assertEqual(early["task_goal"], goal)
        self.assertEqual(later["task_goal"], goal)
        self.assertEqual(later["initial_observation"], initial)
        self.assertEqual(later["recent_interaction"], [steps[-1]])
        self.assertEqual(later["omitted_history_steps"], 1)
        self.assertEqual(later["plan"], plan)
        self.assertEqual(later["experiences"], experiences)

    def test_training_injects_once_per_plan_and_commits_once_per_task(self):
        experiment = self.experiment()
        tasks = [{"id": f"task{i}", "game_path": f"task{i}.json"} for i in range(2)]
        memory = experiment.train(tasks)
        self.assertEqual(memory.data["version"], 2)
        self.assertEqual(self.roles.critiques, 2)
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
