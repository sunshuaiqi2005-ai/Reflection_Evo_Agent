from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experience.cli import compatible_test_selection_update, source_hashes
from experience.data import prepare
from experience.util import digest, read_json


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/main"
        for relative in ("src/experience/episode.py", "src/experience/data.py", "src/experience/cli.py", "prompts/actor.txt"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        for name in ("src", "prompts"):
            shutil.copytree(self.root / name, self.run / "source_snapshot" / name)
        self.previous = {"config": {"seed": 42, "task_selection": {"train_task_types": ["pick_and_place_simple"]}},
                         "model": "fixed", "software": "fixed", "tokenizer_sha256": "fixed",
                         "manifests": {"development.json": "fixed-dev", "plan.json": "fixed-plan", "test.json": "old-test"},
                         "source_hash": digest(source_hashes(self.root))}
        self.current = deepcopy(self.previous)
        self.current["config"]["task_selection"]["test_task_types"] = ["pick_and_place_simple"]
        self.current["manifests"]["test.json"] = "simple-test"
        (self.root / "src/experience/data.py").write_text("test filter changed")
        (self.root / "src/experience/cli.py").write_text("evaluation metadata changed")

    def compatible(self, current=None):
        return compatible_test_selection_update(self.previous, current or self.current, self.root, self.run)

    def test_only_final_test_selector_can_change_during_frozen_training(self):
        self.assertTrue(self.compatible())

    def test_training_plan_config_and_model_changes_are_rejected(self):
        for key in ("development.json", "plan.json"):
            changed = deepcopy(self.current)
            changed["manifests"][key] = "changed"
            self.assertFalse(self.compatible(changed))
        changed = deepcopy(self.current)
        changed["config"]["seed"] = 99
        self.assertFalse(self.compatible(changed))
        changed = deepcopy(self.current)
        changed["model"] = "changed"
        self.assertFalse(self.compatible(changed))

    def test_actor_changes_and_tampered_source_snapshot_are_rejected(self):
        (self.root / "src/experience/episode.py").write_text("changed actor")
        self.assertFalse(self.compatible())
        (self.root / "src/experience/episode.py").write_text("original")
        (self.run / "source_snapshot/prompts/actor.txt").write_text("tampered")
        self.assertFalse(self.compatible())

    def test_simple_test_filter_does_not_change_training_or_extensions(self):
        def catalog(root, split):
            return [{"id": f"{split}/{kind}/{i}", "task_type": kind, "object": str(i),
                     "target": "target", "scene": str(i)}
                    for kind in ("pick_and_place_simple", "pick_heat_then_place_in_recep") for i in range(50)]
        config = {"seed": 42, "data_root": "data/alfworld", "task_selection": {"train_task_types": ["pick_and_place_simple"]},
                  "splits": {"train": 20, "extension": 80, "test": 20, "allow_reduce": True},
                  "evolution": {"rounds": 10, "extension_without_replacement": True}}
        with patch("experience.data.collect", catalog):
            before_root, after_root = self.root / "before", self.root / "after"
            prepare(config, before_root)
            config["task_selection"]["test_task_types"] = ["pick_and_place_simple"]
            summary = prepare(config, after_root)
        for name in ("development.json", "plan.json"):
            self.assertEqual(read_json(before_root / "data/manifests" / name), read_json(after_root / "data/manifests" / name))
        test = read_json(after_root / "data/manifests/test.json")["test"]
        self.assertEqual(len(test), 20)
        self.assertTrue(all(t["task_type"] == "pick_and_place_simple" for t in test))
        self.assertEqual(summary["family_counts"]["test"], {"pick_and_place_simple": 20})


if __name__ == "__main__":
    unittest.main()
