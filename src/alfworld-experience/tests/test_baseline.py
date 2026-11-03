from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experience.memory import Memory
from experience.pipeline import Evolution, evaluate, evaluate_baseline
from experience.util import read_json, write_json


CONFIG = {"seed": 42, "evaluation": {"compare_initial_memory": True, "baseline_before_training": True},
          "evolution": {"trials_per_task": 3, "max_revisions": 1}}


class CachedRunner:
    def __init__(self, interrupt_at=None):
        self.calls, self.cache = [], {}
        self.interrupt_at = interrupt_at

    def run(self, task, memory, seed, episode_id, stage):
        if episode_id in self.cache:
            return deepcopy(self.cache[episode_id])
        if len(self.calls) == self.interrupt_at:
            self.interrupt_at = None
            raise OSError("interrupted baseline")
        self.calls.append({"id": episode_id, "task": task, "memory": memory.to_dict(), "seed": seed})
        trace = {"id": episode_id, "task": task, "success": True, "termination": "won",
                 "initial_observation": "Your task is to: finish the task.", "steps": []}
        self.cache[episode_id] = trace
        return deepcopy(trace)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.tasks = [{"id": f"heldout-{i}", "task_type": "pick_clean_then_place_in_recep"}
                      for i in range(20)]

    def test_twenty_baselines_then_forty_rounds_then_twenty_final_tests(self):
        runner = CachedRunner()
        baseline = evaluate_baseline(CONFIG, self.folder, self.tasks, runner)
        self.assertTrue(baseline["complete"])
        self.assertFalse((self.folder / "memory_final.json").exists())
        self.assertTrue(all(call["memory"]["items"] == [] for call in runner.calls))
        plan = [{"round": i, "original": {"id": f"train-{i}"}, "extension": {"id": f"extension-{i}"}}
                for i in range(1, 41)]
        Evolution(CONFIG, self.folder, runner, object()).evolve(plan)
        result = evaluate(CONFIG, self.folder, self.tasks, runner)
        self.assertEqual(len(runner.calls), 20 + 40 * 3 + 20)
        self.assertEqual([call["id"] for call in runner.calls[:20]],
                         [f"test-{i:03d}-initial" for i in range(1, 21)])
        self.assertTrue(all(call["id"].startswith("r") for call in runner.calls[20:140]))
        self.assertEqual([call["id"] for call in runner.calls[140:]],
                         [f"test-{i:03d}-final" for i in range(1, 21)])
        self.assertEqual([call["seed"] for call in runner.calls[:20]],
                         [call["seed"] for call in runner.calls[140:]])
        self.assertEqual([call["task"] for call in runner.calls[:20]],
                         [call["task"] for call in runner.calls[140:]])
        self.assertEqual(result["outcomes"]["initial"], baseline["outcomes"])
        self.assertEqual(result["metrics"]["initial"], baseline["metrics"])
        self.assertEqual(read_json(self.folder / "status.json")["phase"], "complete")

    def test_interrupted_baseline_resumes_and_completed_baseline_is_not_repeated(self):
        runner = CachedRunner(interrupt_at=2)
        with self.assertRaises(OSError):
            evaluate_baseline(CONFIG, self.folder, self.tasks, runner)
        self.assertEqual(read_json(self.folder / "baseline_progress.json")["completed"], 2)
        self.assertFalse((self.folder / "baseline_results.json").exists())
        result = evaluate_baseline(CONFIG, self.folder, self.tasks, runner)
        self.assertEqual(len(runner.calls), 20)
        write_json(self.folder / "status.json", {"phase": "evolution"})
        self.assertEqual(evaluate_baseline(CONFIG, self.folder, self.tasks, runner), result)
        self.assertEqual(len(runner.calls), 20)
        self.assertEqual(read_json(self.folder / "status.json")["phase"], "evolution")
        with self.assertRaisesRegex(ValueError, "不一致"):
            evaluate_baseline(CONFIG, self.folder, list(reversed(self.tasks)), runner)

    def test_baseline_rejects_nonempty_memory_or_learning_already_started(self):
        nonempty = Memory(1, [{"id": "old", "trigger": "condition", "advice": "action", "exceptions": []}], [])
        write_json(self.folder / "memory_initial.json", nonempty.to_dict())
        with self.assertRaisesRegex(ValueError, "空经验"):
            evaluate_baseline(CONFIG, self.folder, self.tasks, CachedRunner())
        write_json(self.folder / "memory_initial.json", Memory().to_dict())
        write_json(self.folder / "rounds/01/result.json", {})
        with self.assertRaisesRegex(ValueError, "训练已开始"):
            evaluate_baseline(CONFIG, self.folder, self.tasks, CachedRunner())

    def test_final_evaluation_rejects_mismatched_baseline(self):
        runner = CachedRunner()
        evaluate_baseline(CONFIG, self.folder, self.tasks, runner)
        write_json(self.folder / "memory_final.json", {"frozen": True, "memory": Memory().to_dict(),
                                                       "memory_hash": Memory().fingerprint, "rounds_completed": 40})
        with self.assertRaisesRegex(ValueError, "不匹配"):
            evaluate(CONFIG, self.folder, list(reversed(self.tasks)), runner)
        self.assertEqual(len(runner.calls), 20)


if __name__ == "__main__":
    unittest.main()
