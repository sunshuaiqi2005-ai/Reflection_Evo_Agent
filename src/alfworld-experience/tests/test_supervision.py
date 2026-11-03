import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experience.util import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evolution_supervision", ROOT / "scripts/supervise_evolution.py")
supervision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervision)


class SupervisionTests(unittest.TestCase):
    def completed_training(self, run):
        write_json(run / "manifests/plan.json", [{"round": i} for i in range(1, 31)])
        write_json(run / "manifests/summary.json", {"actual": {"test": 20}})
        write_json(run / "run_metadata.json", {"config": {"evaluation": {"compare_initial_memory": True}}})
        write_json(run / "status.json", {"phase": "evolution_complete", "rounds_completed": 30})
        write_json(run / "memory_final.json", {"frozen": True, "memory": {"items": []},
                                               "memory_hash": "frozen-memory", "rounds_completed": 30})

    def completed_test(self):
        return {"tasks": 20, "frozen_memory_hash": "frozen-memory",
                "outcomes": {arm: [{"task_id": str(i)} for i in range(20)] for arm in ("initial", "final")},
                "metrics": {"initial": {"successes": 6, "success_rate": 0.3, "mean_steps": 30.0},
                            "final": {"successes": 10, "success_rate": 0.5, "mean_steps": 20.0}},
                "success_rate_delta": 0.2}

    def test_full_run_requires_all_twenty_tests_in_both_comparison_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            self.assertTrue(supervision.run_complete(run))
            self.assertFalse(supervision.run_complete(run, with_test=True))
            write_json(run / "status.json", {"phase": "complete"})
            result = self.completed_test()
            write_json(run / "test_results.json", result)
            self.assertTrue(supervision.run_complete(run, with_test=True))
            result["outcomes"]["final"].pop()
            write_json(run / "test_results.json", result)
            self.assertFalse(supervision.run_complete(run, with_test=True))
            result = self.completed_test()
            result["frozen_memory_hash"] = "different-memory"
            write_json(run / "test_results.json", result)
            self.assertFalse(supervision.run_complete(run, with_test=True))

    def test_full_run_command_includes_training_and_testing(self):
        run = Path("/temporary/run")
        self.assertEqual(supervision.worker_command(run, with_test=True)[3], "run")
        self.assertEqual(supervision.worker_command(run)[3], "evolve")

    def test_baseline_progress_is_distinct_from_final_testing(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            write_json(run / "status.json", {"phase": "baseline", "completed": 4, "total": 20})
            write_json(run / "baseline_progress.json", {"completed": 4, "total": 20})
            write_json(run / "episodes/test-001-initial.json", {})
            with contextlib.redirect_stdout(io.StringIO()):
                snapshot = supervision.snapshot(run, "running", "started", SimpleNamespace(pid=123),
                                                1800, with_test=True)
            self.assertEqual(snapshot["stage"], "baseline")
            self.assertEqual(snapshot["baseline_tasks_completed"], 4)
            self.assertEqual(snapshot["test_tasks_completed"], 0)
            summary = supervision.summarize(run, "interrupted", "started", 130, with_test=True)
            self.assertFalse(summary["final_test_executed"])
            self.assertFalse(summary["baseline_completed"])

    def test_baseline_first_completion_requires_matching_baseline_results(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            write_json(run / "run_metadata.json", {"config": {"evaluation": {
                "compare_initial_memory": True, "baseline_before_training": True}}})
            write_json(run / "status.json", {"phase": "complete"})
            result = self.completed_test()
            write_json(run / "test_results.json", result)
            self.assertFalse(supervision.run_complete(run, with_test=True))
            baseline = {"complete": True, "tasks": 20, "outcomes": result["outcomes"]["initial"]}
            write_json(run / "baseline_results.json", baseline)
            self.assertTrue(supervision.run_complete(run, with_test=True))
            baseline["outcomes"][0] = {"task_id": "wrong"}
            write_json(run / "baseline_results.json", baseline)
            self.assertFalse(supervision.run_complete(run, with_test=True))

    def test_full_run_completion_uses_forty_round_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            write_json(run / "manifests/plan.json", [{"round": i} for i in range(1, 41)])
            write_json(run / "status.json", {"phase": "complete"})
            write_json(run / "test_results.json", self.completed_test())
            self.assertFalse(supervision.run_complete(run, with_test=True))
            frozen = read_json(run / "memory_final.json")
            frozen["rounds_completed"] = 40
            write_json(run / "memory_final.json", frozen)
            self.assertTrue(supervision.run_complete(run, with_test=True))
            with contextlib.redirect_stdout(io.StringIO()):
                snapshot = supervision.snapshot(run, "complete", "started", SimpleNamespace(pid=123),
                                                1800, with_test=True)
            self.assertEqual(snapshot["rounds_total"], 40)

    def test_monitor_records_testing_progress_and_partial_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            write_json(run / "status.json", {"phase": "test", "completed": 5, "total": 20})
            write_json(run / "test_progress.json", {"completed": 5, "total": 20})
            with contextlib.redirect_stdout(io.StringIO()):
                snapshot = supervision.snapshot(run, "running", "started", SimpleNamespace(pid=123),
                                                1800, with_test=True)
            self.assertEqual(snapshot["stage"], "test")
            self.assertEqual(snapshot["test_tasks_completed"], 5)
            self.assertEqual(snapshot["test_tasks_total"], 20)
            summary = supervision.summarize(run, "interrupted", "started", 130, with_test=True)
            self.assertTrue(summary["final_test_executed"])
            self.assertFalse(summary["final_test_completed"])
            self.assertEqual(summary["test_tasks_completed"], 5)

    def test_completed_summary_includes_frozen_test_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.completed_training(run)
            write_json(run / "status.json", {"phase": "complete"})
            write_json(run / "test_results.json", self.completed_test())
            summary = supervision.summarize(run, "complete", "started", 0, with_test=True)
            self.assertTrue(summary["final_test_executed"])
            self.assertTrue(summary["final_test_completed"])
            self.assertEqual(summary["test_tasks_completed"], 20)
            self.assertEqual(summary["test_metrics"]["final"]["successes"], 10)
            text = (run / "evolution_summary.md").read_text()
            self.assertIn("20/20", text)
            self.assertIn("初始经验", text)
            self.assertIn("最终经验", text)

    def test_snapshot_and_completion_summary_use_all_thirty_planned_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "manifests/plan.json", [{"round": i} for i in range(1, 31)])
            write_json(run / "status.json", {"phase": "evolution_complete", "rounds_completed": 30})
            write_json(run / "memory_final.json", {"memory": {"items": []}, "rounds_completed": 30})
            for i in range(1, 31):
                write_json(run / f"rounds/{i:02d}/result.json", {
                    "round": i, "protocol": "original_once_extension_failure_review_append_v2",
                    "status": "skipped_all_success",
                    "plan": {"original": {"task_type": "pick_heat_then_place_in_recep"}},
                    "episodes": {f"trial-{j}": {"success": True} for j in (1, 2, 3)}})
            with contextlib.redirect_stdout(io.StringIO()):
                snapshot = supervision.snapshot(run, "complete", "started", SimpleNamespace(pid=123), 1800)
            self.assertEqual(snapshot["rounds_total"], 30)
            self.assertEqual(snapshot["rounds_completed"], 30)
            summary = supervision.summarize(run, "complete", "started", 0)
            self.assertEqual(summary["rounds_total"], 30)
            self.assertEqual(summary["rounds_completed"], 30)
            self.assertIn("30/30", (run / "evolution_summary.md").read_text())
            self.assertEqual(read_json(run / "supervision.json")["rounds_total"], 30)

    def test_run_snapshot_plan_takes_priority_over_current_project_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            write_json(root / "data/manifests/plan.json", [{} for _ in range(30)])
            write_json(run / "manifests/plan.json", [{} for _ in range(10)])
            with patch.object(supervision, "ROOT", root):
                self.assertEqual(supervision.planned_rounds(run), 10)
                self.assertEqual(supervision.planned_rounds(root / "new-run"), 30)


if __name__ == "__main__":
    unittest.main()
