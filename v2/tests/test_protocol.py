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
        return {"trajectory_reviews": ["反馈确认两步完成任务。"] * 3,
                "judge_experiences": [{**content(), "phase": phase} for phase in ("planning", "execution")],
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
        self.assertEqual(selected["candidate_count"], 7)
        self.assertFalse(selected["budget_applied"])
        self.assertEqual(self.roles.forecasts, [])

    def test_at_threshold_reads_selected_state_in_full(self):
        memory = self.memory_with(8)
        selected = memory.select("actor", {}, self.roles, 8, 1, "catalogue", self.directory)
        self.assertEqual(selected["mode"], "catalogue")
        self.assertEqual(len(selected["experiences"]), 8)
        self.assertEqual(selected["state_ids"], ["S00001"])
        self.assertEqual(self.roles.forecasts[0][0], "execution")

    def test_selected_state_injects_all_members_within_budget_and_excludes_other_states(self):
        memory = self.memory_with(20)
        extra = memory.add("actor", "execution", content("B"))
        extra["state_id"] = "S00002"
        memory.data["clusters"].append({**memory.data["clusters"][0], "state_id": "S00002",
                                        "state": "B", "experience_ids": [extra["id"]]})
        self.roles.choose = ["S00001"]
        selected = memory.select("actor", {}, self.roles, 8, 1, "select", self.directory)
        self.assertEqual(len(selected["experiences"]), 20)
        self.assertEqual(selected["candidate_count"], 20)
        self.assertFalse(selected["budget_applied"])
        self.assertNotIn(extra["id"], [e["id"] for e in selected["experiences"]])

    def test_oversized_single_state_is_capped_without_new_embeddings(self):
        memory = self.memory_with(25)
        embedded = len(self.roles.client.embedded)
        before = deepcopy(memory.data)

        selected = memory.select("actor", {}, self.roles, 8, 1, "oversized", self.directory)

        self.assertEqual(len(selected["experiences"]), 20)
        self.assertEqual(selected["candidate_count"], 25)
        self.assertTrue(selected["budget_applied"])
        self.assertEqual(selected["state_ids"], ["S00001"])
        self.assertEqual([e["id"] for e in selected["experiences"]],
                         [e["id"] for e in memory.data["experiences"][:20]])
        self.assertEqual(len(self.roles.client.embedded), embedded)
        self.assertEqual(memory.data, before)

    def test_budget_round_robins_ranked_states_and_skips_exhausted_groups(self):
        memory = self.memory_with(21)
        groups = {"S00001": list(memory.data["experiences"])}
        for state_id, state, count in (("S00002", "B", 2), ("S00003", "C", 1)):
            entries = [memory.add("actor", "execution", content(state)) for _ in range(count)]
            for entry in entries:
                entry["state_id"] = state_id
            memory.data["clusters"].append({"state_id": state_id, "type": "actor", "phase": "execution",
                "state": state, "experience_ids": [e["id"] for e in entries], "centroid": [0., 1.]})
            groups[state_id] = entries
        self.roles.choose = ["S00003", "S00002", "S00001"]

        selected = memory.select("actor", {}, self.roles, 8, 1, "round-robin", self.directory)

        first, second, third = groups["S00001"], groups["S00002"], groups["S00003"]
        expected = [third[0]["id"], second[0]["id"], first[0]["id"], second[1]["id"]]
        expected.extend(e["id"] for e in first[1:17])
        self.assertEqual([e["id"] for e in selected["experiences"]], expected)
        self.assertEqual(selected["state_ids"], self.roles.choose)
        self.assertEqual(selected["candidate_count"], 24)
        self.assertTrue(selected["budget_applied"])

    def test_selection_within_budget_preserves_all_members_in_global_order(self):
        memory = self.memory_with(4)
        entries = [memory.add("actor", "execution", content("B")) for _ in range(4)]
        for entry in entries:
            entry["state_id"] = "S00002"
        memory.data["clusters"].append({"state_id": "S00002", "type": "actor", "phase": "execution",
            "state": "B", "experience_ids": [e["id"] for e in entries], "centroid": [0., 1.]})
        self.roles.choose = ["S00002", "S00001"]

        selected = memory.select("actor", {}, self.roles, 8, 1, "uncapped", self.directory)

        self.assertEqual([e["id"] for e in selected["experiences"]],
                         [e["id"] for e in memory.data["experiences"]])
        self.assertEqual(selected["candidate_count"], 8)
        self.assertFalse(selected["budget_applied"])

    def test_configurable_budget_applies_to_full_and_catalogue_selection(self):
        for count, budget in ((7, 3), (17, 4), (17, 20)):
            with self.subTest(count=count, budget=budget):
                memory = self.memory_with(count)
                forecasts = len(self.roles.forecasts)

                selected = memory.select("actor", {}, self.roles, 8, 1, f"budget-{count}-{budget}",
                                         self.directory, max_experiences=budget)

                self.assertEqual(len(selected["experiences"]), min(count, budget))
                self.assertEqual(selected["candidate_count"], count)
                self.assertEqual(selected["budget_applied"], count > budget)
                self.assertEqual(len(self.roles.forecasts) - forecasts, int(count >= 8))

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

    def test_exact_duplicates_within_and_across_rounds_do_not_allocate_ids_or_reweight_centroid(self):
        first, second = content("A"), content("near")
        reordered = dict(reversed(list(first.items())))
        critique = {"judge_experiences": [], "planner_experiences": [],
                    "actor_experiences": [first, reordered, second, deepcopy(second)]}

        memory = Memory().updated(critique, self.roles.client, .75)

        self.assertEqual([e["id"] for e in memory.data["experiences"]], ["E00001", "E00002"])
        self.assertEqual(memory.data["next_experience"], 3)
        self.assertEqual(memory.data["clusters"][0]["experience_ids"], ["E00001", "E00002"])
        center = np.asarray([[1., 0.], [.8, .6]]).mean(axis=0)
        np.testing.assert_allclose(memory.data["clusters"][0]["centroid"], center / np.linalg.norm(center))
        before = deepcopy(memory.data)
        embedding_calls = len(self.roles.client.embedded)

        after = memory.updated(critique, self.roles.client, .75)

        self.assertEqual(memory.data, before)
        self.assertEqual(after.data, {**before, "version": 2})
        self.assertEqual(len(self.roles.client.embedded), embedding_calls)

    def test_same_state_with_different_body_fields_is_preserved(self):
        fresh = [content("A"), {**content("A"), "explanation": "另一个解释"},
                 {**content("A"), "advice": ["另一种做法"]},
                 {**content("A"), "exceptions": ["另一种例外"]}]
        after = Memory().updated({"judge_experiences": [], "planner_experiences": [],
                                  "actor_experiences": fresh}, self.roles.client, .75)

        self.assertEqual(len(after.data["clusters"]), 1)
        self.assertEqual([{k: e[k] for k in content()} for e in after.data["experiences"]], fresh)

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
        config = {"seed": 42, "data_root": "data", "episode": {"max_steps": 2, "history_tokens": 3200},
                  "memory": {"full_threshold": 1, "similarity_threshold": .8, "max_injected_experiences": 20}}
        return Experiment(config, self.directory, self.directory, self.roles, FakeEnvironment)

    def test_experiment_forwards_configured_experience_budget(self):
        experiment = self.experiment()
        experiment.config["memory"]["full_threshold"] = 8
        experiment.config["memory"]["max_injected_experiences"] = 3

        selected = experiment.injection(self.memory_with(7), "actor", {}, 1, "configured-budget")

        self.assertEqual(selected["mode"], "full")
        self.assertEqual(len(selected["experiences"]), 3)
        self.assertEqual(selected["candidate_count"], 7)
        self.assertTrue(selected["budget_applied"])
        self.assertEqual(self.roles.forecasts, [])

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
        self.assertEqual(set(payload), {"task_goal", "initial_observation", "plans", "step_fields", "trajectories"})
        review_schema = client.chat.call_args.args[2]["properties"]["trajectory_reviews"]
        self.assertEqual(review_schema, {"type": "array", "items": {"type": "string"},
                                         "minItems": 3, "maxItems": 3})

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
        rounds = [read_json(p) for p in sorted((self.directory / "rounds").glob("*.json"))]
        self.assertTrue(all(r["critique"]["trajectory_reviews"] == ["反馈确认两步完成任务。"] * 3 for r in rounds))
        self.assertEqual(len(memory.data["experiences"]), 4)
        self.assertTrue(all(set(e) == {"id", "type", "phase", "state_id", "state", "explanation", "advice", "exceptions"}
                            for e in memory.data["experiences"]))

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
