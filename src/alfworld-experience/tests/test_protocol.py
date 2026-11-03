"""Deterministic protocol tests use scripted doubles, never benchmark measurements."""
from copy import deepcopy
import hashlib
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experience.data import choose_extension, prepare, random_sample, stratified_sample
from experience.episode import EpisodeRunner
from experience.memory import Memory, experience_with_id
from experience.pipeline import Evolution, choose_pair, evaluate
from experience.util import ContextBudgetError, InfrastructureError, ProtocolError, read_json, write_json


CONFIG = {"seed": 42, "evolution": {"trials_per_task": 3, "max_revisions": 1},
          "evaluation": {"compare_initial_memory": True}, "data_root": "data",
          "episode": {"max_steps": 2, "history_tokens": 1000},
          "model": {"actor_context": 10000, "actor_output_tokens": 64}}
TASK = {"id": "train/original", "task_type": "pick_and_place_simple"}
EXTENSION = {"id": "valid_seen/extension", "task_type": "pick_and_place_simple"}
PLAN = {"round": 1, "original": TASK, "extension": EXTENSION, "extension_seed": 123}


def trace(identifier, task=TASK, success=False):
    return {"id": identifier, "task": task, "success": success, "completed": True,
            "initial_observation": "Find and place the object.", "termination": "won" if success else "step_limit",
            "steps": [{"step": 1, "action": "open cabinet 1", "observation": "The cabinet is empty.",
                       "score": int(success), "won": success}]}


def rule(t, revised=False):
    return {"trigger": "a closed container must be searched" if not revised else "an already inspected container is unchanged",
            "advice": "Open it before checking its contents." if not revised else "Search another uninspected location.",
            "exceptions": ["Recheck if the environment state has changed."]}


class ScriptedRunner:
    def __init__(self, successes):
        self.successes, self.calls = iter(successes), []

    def run(self, task, memory, actor_seed, episode_id, stage):
        self.calls.append({"id": episode_id, "task": deepcopy(task), "seed": actor_seed,
                           "memory": memory.to_dict(), "memory_hash": memory.fingerprint, "stage": stage})
        return trace(episode_id, task, next(self.successes))


class ScriptedCognition:
    def __init__(self, revise=False):
        self.revise, self.calls = revise, []

    def propose(self, memory, pair, index):
        self.calls.append("propose")
        self.pair = pair
        return {"analysis": "Observed container inspection.", "candidate": rule(pair[0])}

    def review(self, memory, candidate, traces, index):
        self.calls.append("review")
        return {"trigger": "present", "followed": "yes", "contribution": "uncertain",
                "needs_revision": self.revise, "issue": "Missing unchanged-state condition." if self.revise else "",
                "revision": rule(traces[-1], True) if self.revise else None}

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_round(self, flags, revise=False, memory=None):
        runner, cognition = ScriptedRunner(flags), ScriptedCognition(revise)
        original = memory or Memory()
        evolved, result = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, original)
        return evolved, result, runner, cognition

    def test_three_trials_share_frozen_memory_but_distinct_seeds(self):
        memory, result, runner, _ = self.run_round([True, False, False, False])
        self.assertEqual(len({c["seed"] for c in runner.calls[:3]}), 3)
        self.assertEqual(len({c["memory_hash"] for c in runner.calls[:3]}), 1)
        self.assertEqual(result["status"], "rejected_original")
        self.assertEqual(memory.fingerprint, Memory().fingerprint)
        self.assertEqual(len(runner.calls), 4)

    def test_all_success_skips_reflection_and_extension(self):
        memory, result, runner, cognition = self.run_round([True, True, True])
        self.assertEqual(result["status"], "skipped_all_success")
        self.assertEqual(cognition.calls, [])
        self.assertEqual(len(runner.calls), 3)

    def test_failure_pair_prefers_different_behavior(self):
        traces = [trace("a"), trace("b"), trace("c")]
        traces[2]["steps"][0]["action"] = "look"
        pair, mode = choose_pair(traces, 42)
        self.assertEqual(mode, "failure_failure")
        self.assertIn("c", [t["id"] for t in pair])
        self.assertTrue(all(not t["success"] for t in pair))

    def test_mixed_pair_is_seeded_success_failure(self):
        traces = [trace("a", success=True), trace("b"), trace("c")]
        first, mode = choose_pair(traces, 19)
        self.assertEqual(first, choose_pair(traces, 19)[0])
        self.assertEqual([t["success"] for t in first], [True, False])

    def test_extension_failure_does_not_automatically_revise(self):
        memory, result, runner, cognition = self.run_round([True, False, False, True, False])
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(cognition.calls, ["propose", "review"])
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(memory.items[0]["validated_by"], ["r01-candidate-original"])
        self.assertFalse(memory.items[0]["acceptance"]["extension_success"])
        self.assertFalse(memory.items[0]["acceptance"]["post_review_revalidated"])
        self.assertTrue(memory.items[0]["acceptance"]["extension_review_performed"])

    def test_one_revision_is_appended_without_any_more_episodes(self):
        memory, result, runner, cognition = self.run_round(
            [False, False, False, True, False], revise=True)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(memory.version, 1)
        self.assertEqual(memory.items[0]["id"], "r01-revised")
        self.assertEqual(memory.items[0]["validated_by"], [])
        self.assertEqual(memory.items[0]["reviewed_on"], ["r01-candidate-extension"])
        self.assertEqual(cognition.calls, ["propose", "review"])
        self.assertEqual([c["stage"] for c in runner.calls],
                         ["trial", "trial", "trial", "candidate-original", "candidate-extension"])

    def test_successful_extension_is_appended_unchanged_without_review(self):
        memory, result, runner, cognition = self.run_round(
            [False, False, False, True, True], revise=True)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(cognition.calls, ["propose"])
        self.assertTrue(memory.items[0]["acceptance"]["extension_success"])
        self.assertFalse(memory.items[0]["acceptance"]["extension_review_performed"])
        self.assertFalse(memory.items[0]["acceptance"]["revised"])
        self.assertEqual(memory.items[0]["id"], "r01-candidate")
        self.assertEqual(memory.items[0]["validated_by"], ["r01-candidate-original", "r01-candidate-extension"])
        self.assertEqual(memory.items[0]["reviewed_on"], [])
        for key in ("trigger", "advice", "exceptions"):
            self.assertEqual(memory.items[0][key], rule(trace("a"))[key])
        self.assertFalse((self.folder / "rounds/01/extension_review.json").exists())
        self.assertNotIn("revised_candidate", result)

    def test_successful_extension_does_not_need_an_available_review_model(self):
        cognition = ScriptedCognition()
        def unavailable(*args):
            raise InfrastructureError("Review model unavailable; successful extensions must skip it")
        cognition.review = unavailable
        runner = ScriptedRunner([True, False, False, True, True])
        memory, result = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(memory.version, 1)
        self.assertEqual(len(runner.calls), 5)

    def test_original_failure_never_runs_extension_or_review(self):
        memory, result, runner, cognition = self.run_round([True, False, False, False], revise=True)
        self.assertEqual(result["status"], "rejected_original")
        self.assertEqual(cognition.calls, ["propose"])
        self.assertEqual(len(runner.calls), 4)
        self.assertNotIn("candidate-extension", result["episodes"])
        self.assertFalse(memory.items)

    def test_round_results_from_old_review_policy_cannot_be_reused(self):
        memory, result, runner, cognition = self.run_round([True, False, False, True, True])
        result["protocol"] = "original_once_extension_once_append"
        write_json(self.folder / "rounds/01/result.json", result)
        calls = len(runner.calls)
        with self.assertRaisesRegex(RuntimeError, "resume inputs"):
            Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(len(runner.calls), calls)

    def test_appending_preserves_old_memory(self):
        old = experience_with_id(rule(trace("old-trace")), "old")
        base = Memory(2, [old], [{"source_ids": ["old"], "relation": "independent"}])
        before = base.to_dict()
        memory, result, runner, _ = self.run_round([True, False, False, True, False], revise=True, memory=base)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(base.to_dict(), before)
        self.assertEqual(memory.items[0], old)
        self.assertEqual(len(memory.items), 2)
        self.assertEqual(memory.version, 3)

    def test_passing_candidate_commits_once_and_resumes(self):
        memory, result, runner, cognition = self.run_round([True, False, False, True, True])
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(memory.version, 1)
        calls = len(runner.calls)
        resumed, _ = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(resumed.fingerprint, memory.fingerprint)
        self.assertEqual(len(runner.calls), calls)

    def test_duplicate_experience_id_cannot_be_appended(self):
        candidate = experience_with_id(rule(trace("a")), "candidate")
        base = Memory().appended(candidate)
        with self.assertRaises(ProtocolError):
            base.appended(candidate)

    def test_source_conditions_and_exceptions_survive_append(self):
        old = experience_with_id(rule(trace("a")), "old")
        new = experience_with_id(rule(trace("b"), True), "new")
        base = Memory(1, [old], [{"source_ids": ["old"], "relation": "independent"}])
        appended = base.appended(new)
        self.assertEqual(appended.items[0], old)
        self.assertIn(old["exceptions"][0], appended.render())
        self.assertIn(new["advice"], appended.render())

    def test_concrete_candidate_content_reaches_validation_and_is_preserved(self):
        candidates = [
            {"trigger": "searching for an apple", "advice": "Check the apple on the countertop.", "exceptions": []},
            {"trigger": "taking a pan", "advice": "Check the pan before taking it.", "exceptions": []},
            {"trigger": "at cabinet 3 in FloorPlan301", "advice": "Search cabinet 3; trial_T20190909.",
             "exceptions": ["Leave bowl 2 in drawer 1."]},
        ]
        for index, candidate in enumerate(candidates):
            with self.subTest(candidate=candidate):
                plan = deepcopy(PLAN)
                plan["original"]["object"] = "Apple" if index == 0 else "Pan"
                runner, cognition = ScriptedRunner([False, False, False, True, True]), ScriptedCognition()
                before = deepcopy(candidate)
                with patch.object(cognition, "propose", return_value={"analysis": "", "candidate": candidate}):
                    memory, result = Evolution(CONFIG, self.folder / str(index), runner, cognition).round(plan, Memory())
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(len(runner.calls), 5)
                self.assertEqual(candidate, before)
                for field in ("trigger", "advice", "exceptions"):
                    self.assertEqual(memory.items[0][field], candidate[field])
                    self.assertEqual(runner.calls[3]["memory"]["items"][0][field], candidate[field])

    def test_revised_content_is_appended_without_filtering_or_reexecution(self):
        revised = {"trigger": "apple in cabinet 3", "advice": "Check apple in FloorPlan301.",
                   "exceptions": ["trial_T20190909"]}
        runner, cognition = ScriptedRunner([False, False, False, True, False]), ScriptedCognition()
        review = {"needs_revision": True, "issue": "Clarify the next action.", "revision": revised}
        with patch.object(cognition, "review", return_value=review):
            memory, result = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(memory.items[0]["id"], "r01-revised")
        for field in ("trigger", "advice", "exceptions"):
            self.assertEqual(memory.items[0][field], revised[field])
        self.assertFalse(memory.items[0]["acceptance"]["post_review_revalidated"])

    def test_infrastructure_failure_does_not_become_rejected_task(self):
        class BrokenRunner:
            def run(self, *args):
                raise InfrastructureError("API unavailable")
        with self.assertRaises(InfrastructureError):
            Evolution(CONFIG, self.folder, BrokenRunner(), ScriptedCognition()).round(PLAN, Memory())
        self.assertFalse((self.folder / "rounds/01/result.json").exists())

    def test_final_test_requires_freeze_and_never_updates_it(self):
        frozen = {"frozen": True, "memory": Memory().to_dict(), "memory_hash": Memory().fingerprint}
        write_json(self.folder / "memory_final.json", frozen)
        write_json(self.folder / "memory_initial.json", Memory().to_dict())
        runner = ScriptedRunner([True, False, False, True])
        tasks = [dict(TASK, id="test/a"), dict(TASK, id="test/b")]
        result = evaluate(CONFIG, self.folder, tasks, runner)
        self.assertEqual(read_json(self.folder / "memory_final.json"), frozen)
        self.assertEqual(result["tasks"], 2)
        for task in tasks:
            seeds = [c["seed"] for c in runner.calls if c["task"]["id"] == task["id"]]
            self.assertEqual(len(set(seeds)), 1)

    def test_stratified_sampling_and_extension_draw(self):
        tasks = [{"id": f"{family}/{i}", "task_type": family, "object": str(i), "target": "x", "scene": str(i)}
                 for family in ("a", "b") for i in range(4)]
        sample = stratified_sample(tasks, 4, 42)
        self.assertEqual(sample, stratified_sample(tasks, 4, 42))
        self.assertEqual(sum(t["task_type"] == "a" for t in sample), 2)
        chosen = choose_extension(tasks[0], tasks[1:], 42)
        self.assertEqual(chosen["task_type"], "a")
        self.assertNotEqual(chosen["object"], tasks[0]["object"])

    def test_training_sampling_is_uniform_seeded_and_without_replacement(self):
        tasks = [{"id": str(i), "task_type": "majority" if i < 19 else "minority"} for i in range(20)]
        selected = random_sample(tasks, 10, 42)
        expected = random.Random(42).sample(sorted(tasks, key=lambda t: t["id"]), 10)
        self.assertEqual(selected, expected)
        self.assertEqual(len({t["id"] for t in selected}), 10)
        self.assertEqual(selected, random_sample(list(reversed(tasks)), 10, 42))
        self.assertNotEqual(selected, random_sample(tasks, 10, 43))

    def test_small_dataset_reduces_counts_explicitly(self):
        counts = {"train": 3, "valid_seen": 6, "valid_unseen": 2}
        def catalog(root, split):
            return [{"id": f"{split}/{i}", "task_type": "one_family", "object": str(i),
                     "target": "target", "scene": str(i)} for i in range(counts[split])]
        config = deepcopy(CONFIG)
        config["evolution"].update({"rounds": 10, "extension_without_replacement": True})
        config["splits"] = {"train": 20, "extension": 80, "test": 20, "allow_reduce": True}
        with patch("experience.data.collect", catalog):
            summary = prepare(config, self.folder)
        self.assertEqual(summary["actual"], {"train": 3, "extension": 6, "test": 2})
        self.assertEqual(summary["rounds"], 3)
        self.assertTrue(summary["warnings"])
        plan = read_json(self.folder / "data/manifests/plan.json")
        self.assertEqual(len({p["extension"]["id"] for p in plan}), 3)
        self.assertEqual(summary["sampling"]["train_pool"], "uniform_without_replacement")
        self.assertEqual(summary["sampling"]["round_originals"], "uniform_without_replacement")

    def test_not_following_advice_can_be_clarified_without_extra_episodes(self):
        cognition = ScriptedCognition(revise=True)
        original_review = cognition.review
        def unsupported(*args):
            result = original_review(*args)
            result["followed"] = "no"
            return result
        cognition.review = unsupported
        runner = ScriptedRunner([True, False, False, True, False])
        memory, result = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(memory.items[0]["id"], "r01-revised")

    def test_revision_without_explanation_is_rejected_without_extra_episodes(self):
        cognition = ScriptedCognition(revise=True)
        original_review = cognition.review
        def unsupported(*args):
            result = original_review(*args)
            result["issue"] = ""
            return result
        cognition.review = unsupported
        runner = ScriptedRunner([True, False, False, True, False])
        memory, result = Evolution(CONFIG, self.folder, runner, cognition).round(PLAN, Memory())
        self.assertEqual(result["status"], "rejected_protocol")
        self.assertEqual(len(runner.calls), 5)
        self.assertFalse(memory.items)


class EpisodeTests(unittest.TestCase):
    def test_context_budget_never_discards_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "prompts").mkdir()
            (project / "prompts/actor.txt").write_text("{memory}")
            class Client:
                def tokens(self, text): return len(text)
                def chat_tokens(self, messages): return sum(len(m["content"]) for m in messages)
            config = deepcopy(CONFIG)
            config["model"]["actor_context"] = 100
            memory = Memory(1, [{"id": "old", "trigger": "condition", "advice": "X" * 500, "exceptions": []}])
            before = memory.fingerprint
            runner = EpisodeRunner(config, project, project / "runs", Client())
            with self.assertRaises(ContextBudgetError):
                runner.messages(memory, "Original task", [])
            self.assertEqual(memory.fingerprint, before)

    def test_interrupted_episode_replays_before_continuing(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "prompts").mkdir()
            (project / "prompts/actor.txt").write_text("{memory}")
            (project / "data").mkdir()
            game = project / "data/game"
            game.write_text("fixture")
            task = {"id": "fixture", "task_type": "fixture", "game_path": "game",
                    "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest()}
            class Client:
                calls = 0
                def tokens(self, text): return len(text)
                def chat_tokens(self, messages): return sum(len(m["content"]) for m in messages)
                def call(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 2:
                        raise InfrastructureError("interrupted")
                    return {"text": "look", "call_hash": "fixture"}
            resets = []
            class Environment:
                def __init__(self, file, seed): resets.append(seed)
                def reset(self): return "Task", False
                def step(self, action): return "Nothing happens.", 0.0, False, False
                def close(self): pass
            runner = EpisodeRunner(CONFIG, project, project / "runs", Client(), Environment)
            with self.assertRaises(InfrastructureError):
                runner.run(task, Memory(), 1, "a", "trial")
            self.assertTrue((project / "runs/episodes/a.progress.json").exists())
            result = runner.run(task, Memory(), 1, "a", "trial")
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["steps"]), 2)
            self.assertEqual(resets[0], resets[1])
            self.assertFalse((project / "runs/episodes/a.progress.json").exists())

    def test_success_comes_from_environment_and_reset_is_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "prompts").mkdir()
            (project / "prompts/actor.txt").write_text("Solve the task. {memory}")
            (project / "data").mkdir()
            game = project / "data/game"
            game.write_text("fake fixture, not a benchmark environment")
            task = {"id": "fixture", "task_type": "fixture", "game_path": "game",
                    "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest()}
            class Client:
                def tokens(self, text): return len(text)
                def chat_tokens(self, messages): return sum(len(m["content"]) for m in messages)
                def call(self, *args, **kwargs): return {"text": "Task completed successfully!", "call_hash": "fixture"}
            resets = []
            class Environment:
                def __init__(self, file, seed): resets.append(seed)
                def reset(self): return "Your task: find an object.", False
                def step(self, action): return "Nothing happens.", 0.0, False, False
                def close(self): pass
            runner = EpisodeRunner(CONFIG, project, project / "runs", Client(), Environment)
            first = runner.run(task, Memory(), 1, "a", "trial")
            second = runner.run(task, Memory(), 2, "b", "trial")
            self.assertFalse(first["success"])
            self.assertEqual(first["termination"], "step_limit")
            self.assertEqual(first["initial_observation"], second["initial_observation"])
            self.assertEqual(resets[0], resets[1])
            with self.assertRaises(InfrastructureError):
                runner.run(task, Memory(), 99, "a", "trial")


if __name__ == "__main__":
    unittest.main()
