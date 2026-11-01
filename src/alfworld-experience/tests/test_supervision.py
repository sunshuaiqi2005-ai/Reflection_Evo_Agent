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
