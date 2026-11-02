from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experience.data import prepare, random_sample, stratified_sample
from experience.episode import normalize_action
from experience.memory import validate_candidate
from experience.util import ProtocolError, read_json, seed_for
from experience.visible_state import visible_state, render_visible_state


class ImprovementTests(unittest.TestCase):
    def test_legacy_placement_maps_to_native_textworld_without_changing_target(self):
        self.assertEqual(normalize_action("put cup 1 in/on cabinet 1"), "move cup 1 to cabinet 1")
        self.assertEqual(normalize_action("Action: put apple 2 in/on fridge 1"), "move apple 2 to fridge 1")
        self.assertEqual(normalize_action("move cup 1 to cabinet 1"), "move cup 1 to cabinet 1")
        self.assertEqual(normalize_action("take cup 1 from fridge 1"), "take cup 1 from fridge 1")

    def test_failed_actions_do_not_change_location_or_held_item(self):
        steps = [{"action": "go to sinkbasin 1", "observation": "You arrive at sinkbasin 1. On the sinkbasin 1, you see a egg 1."},
                 {"action": "take egg 1 from sinkbasin 1", "observation": "You pick up the egg 1 from the sinkbasin 1."},
                 {"action": "go to fridge 1", "observation": "You arrive at fridge 1. The fridge 1 is open. In it, you see a cup 1."}]
        for _ in range(20):
            steps.append({"action": "take cup 1 from fridge 1", "observation": "Nothing happens."})
        state = visible_state(steps)
        self.assertEqual(state["held_object"], "egg 1")
        self.assertEqual(state["current_location"], "fridge 1")
        self.assertEqual(state["consecutive_ineffective_repeats"], 20)
        self.assertIn("egg 1", render_visible_state(state, len, 550))
        self.assertEqual(visible_state([])["visited_locations"], [])

    def test_successful_placement_frees_the_hands(self):
        state = visible_state([
            {"action": "take cup 1 from fridge 1", "observation": "You pick up the cup 1 from the fridge 1."},
            {"action": "move cup 1 to cabinet 1", "observation": "You move the cup 1 to the cabinet 1."}])
        self.assertEqual(state["held_object"], "empty")

    def test_inventory_feedback_is_grounded_state(self):
        state = visible_state([{"action": "inventory", "observation": "You are carrying: a cup 1."}])
        self.assertIn("cup 1", state["held_object"])
        state = visible_state([{"action": "inventory", "observation": "You are not carrying anything."}])
        self.assertEqual(state["held_object"], "empty")

    def test_candidate_cannot_memorize_concrete_target_type(self):
        trace = {"id": "a", "task": {"object": "Cup"}, "initial_observation": "A cup is visible.",
                 "steps": []}
        rule = {"trigger": "observing a cup in the fridge", "advice": "take the cup from the fridge",
                "exceptions": []}
        with self.assertRaises(ProtocolError):
            validate_candidate(rule, [trace], "bad")
        rule["trigger"] = "the target is visible and the hands are occupied"
        rule["advice"] = "Place the unrelated held object in a compatible receptacle before taking the target."
        self.assertEqual(validate_candidate(rule, [trace], "good")["id"], "good")

    def test_simple_training_selection_preserves_test_sample(self):
        def catalog(root, split):
            return [{"id": f"{split}/{kind}/{i}", "task_type": kind, "object": str(i),
                     "target": "target", "scene": str(i)}
                    for kind in ("pick_and_place_simple", "pick_heat_then_place_in_recep") for i in range(50)]
        config = {"seed": 42, "data_root": "data/alfworld", "task_selection": {"train_task_types": ["pick_and_place_simple"]},
                  "splits": {"train": 20, "extension": 80, "test": 20, "allow_reduce": True},
                  "evolution": {"rounds": 10, "extension_without_replacement": True}}
        with tempfile.TemporaryDirectory() as temp, patch("experience.data.collect", catalog):
            project = Path(temp)
            summary = prepare(config, project)
            plan = read_json(project / "data/manifests/plan.json")
            self.assertEqual(summary["family_counts"]["train"], {"pick_and_place_simple": 20})
            self.assertTrue(all(row["original"]["task_type"] == row["extension"]["task_type"] == "pick_and_place_simple" for row in plan))
            test = read_json(project / "data/manifests/test.json")["test"]
            expected = stratified_sample(catalog(None, "valid_unseen"), 20, seed_for(42, "split", "test"))
            self.assertEqual(test, expected)
            self.assertEqual(summary["actual"], {"train": 20, "extension": 80, "test": 20})

    def test_processing_curriculum_filters_all_pools_and_keeps_matching_extensions(self):
        kinds = ["pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                 "pick_cool_then_place_in_recep"]
        def catalog(root, split):
            return [{"id": f"{split}/{kind}/{i}", "task_type": kind, "object": str(i),
                     "target": "target", "scene": str(i)}
                    for kind in kinds + ["pick_and_place_simple", "pick_two_obj_and_place"]
                    for i in range(30)]
        config = {"seed": 42, "data_root": "data/alfworld",
                  "task_selection": {name + "_task_types": kinds for name in ("train", "extension", "test")},
                  "splits": {"train": 20, "extension": 80, "test": 20, "allow_reduce": True},
                  "evolution": {"rounds": 10, "extension_without_replacement": True}}
        with tempfile.TemporaryDirectory() as temp, patch("experience.data.collect", catalog):
            project = Path(temp)
            summary = prepare(config, project)
            self.assertEqual(summary["actual"], {"train": 20, "extension": 80, "test": 20})
            for counts in summary["family_counts"].values():
                self.assertEqual(set(counts), set(kinds))
            plan = read_json(project / "data/manifests/plan.json")
            self.assertEqual(len(plan), 10)
            self.assertEqual(len({row["original"]["id"] for row in plan}), 10)
            self.assertEqual(len({row["extension"]["id"] for row in plan}), 10)
            self.assertTrue(all(row["original"]["task_type"] == row["extension"]["task_type"] for row in plan))
            development = read_json(project / "data/manifests/development.json")
            test = read_json(project / "data/manifests/test.json")["test"]
            ids = [task["id"] for pool in [development["train"], development["extension"], test] for task in pool]
            self.assertEqual(len(set(ids)), 120)
            self.assertEqual(prepare(config, project), summary)

    def test_extension_filter_rejects_unsupported_task_types(self):
        config = {"data_root": "data/alfworld", "task_selection": {"extension_task_types": ["unknown"]}}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "extension_task_types"):
                prepare(config, Path(temp))

    def test_extension_transfer_preserves_base_training_and_heldout_tasks(self):
        kinds = ["pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                 "pick_cool_then_place_in_recep"]
        def catalog(root, split):
            counts = [22, 23, 23] if split == "valid_seen" else [30, 30, 30]
            return [{"id": f"{split}/{kind}/{i}", "official_split": split, "task_type": kind,
                     "object": str(i), "target": "target", "scene": str(i)}
                    for kind, count in zip(kinds, counts) for i in range(count)]
        config = {"seed": 42, "data_root": "data/alfworld",
                  "task_selection": {name + "_task_types": kinds for name in ("train", "extension", "test")},
                  "splits": {"train": 20, "extension": 80, "test": 20, "allow_reduce": True},
                  "evolution": {"rounds": 10, "extension_without_replacement": True}}
        with tempfile.TemporaryDirectory() as temp, patch("experience.data.collect", catalog):
            before, after = Path(temp) / "before", Path(temp) / "after"
            prepare(config, before)
            old = read_json(before / "data/manifests/development.json")
            new_config = deepcopy(config)
            new_config["splits"].update(train=30, extension=58)
            new_config["pool_transfer"] = {"extension_to_train": 10}
            summary = prepare(new_config, after)
            new = read_json(after / "data/manifests/development.json")
            self.assertEqual(summary["actual"], {"train": 30, "extension": 58, "test": 20})
            self.assertEqual(new["train"][:20], old["train"])
            expected = random_sample(old["extension"], 10, seed_for(42, "extension-to-train"))
            self.assertEqual(new["train"][20:], expected)
            self.assertTrue(all(t["official_split"] == "valid_seen" for t in expected))
            self.assertEqual({t["id"] for t in new["extension"]} | {t["id"] for t in expected},
                             {t["id"] for t in old["extension"]})
            self.assertFalse({t["id"] for t in new["train"]} & {t["id"] for t in new["extension"]})
            self.assertEqual(read_json(before / "data/manifests/test.json"),
                             read_json(after / "data/manifests/test.json"))
            self.assertEqual(summary["pool_transfer"]["task_ids"], [t["id"] for t in expected])
            self.assertEqual(prepare(new_config, after), summary)
            plan = read_json(after / "data/manifests/plan.json")
            self.assertEqual(len(plan), 10)
            self.assertTrue(all(row["original"]["task_type"] == row["extension"]["task_type"] for row in plan))

    def test_extension_transfer_rejects_invalid_counts(self):
        config = {"data_root": "data", "splits": {"train": 30}}
        with tempfile.TemporaryDirectory() as temp:
            for count in (-1, 1.5, True, 31):
                config["pool_transfer"] = {"extension_to_train": count}
                with self.assertRaisesRegex(ValueError, "extension_to_train"):
                    prepare(config, Path(temp))

    def test_manifest_order_runs_all_thirty_training_tasks_once(self):
        def catalog(root, split):
            count = 68 if split == "valid_seen" else 50
            return [{"id": f"{split}/{i}", "official_split": split,
                     "task_type": "pick_heat_then_place_in_recep", "object": str(i),
                     "target": "target", "scene": str(i)} for i in range(count)]
        config = {"seed": 42, "data_root": "data/alfworld",
                  "splits": {"train": 30, "extension": 58, "test": 20, "allow_reduce": True},
                  "pool_transfer": {"extension_to_train": 10},
                  "evolution": {"rounds": 30, "original_order": "manifest",
                                "extension_without_replacement": True}}
        with tempfile.TemporaryDirectory() as temp, patch("experience.data.collect", catalog):
            project = Path(temp)
            summary = prepare(config, project)
            development = read_json(project / "data/manifests/development.json")
            plan = read_json(project / "data/manifests/plan.json")
            self.assertEqual(summary["rounds"], 30)
            self.assertEqual(summary["sampling"]["round_originals"], "manifest_order_without_replacement")
            self.assertEqual([row["original"] for row in plan], development["train"])
            self.assertEqual([row["round"] for row in plan], list(range(1, 31)))
            self.assertEqual(len({row["original"]["id"] for row in plan}), 30)
            self.assertEqual(len({row["extension"]["id"] for row in plan}), 30)
            self.assertTrue(all(row["original"]["task_type"] == row["extension"]["task_type"] for row in plan))
            self.assertEqual(sum(t["official_split"] == "valid_seen" for t in development["train"]), 10)
            self.assertEqual(prepare(config, project), summary)

    def test_invalid_original_order_is_rejected(self):
        config = {"data_root": "data", "splits": {"train": 30},
                  "evolution": {"original_order": "unsupported"}}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "original_order"):
                prepare(config, Path(temp))

    def test_second_transfer_appends_ten_tasks_and_preserves_existing_thirty(self):
        def catalog(root, split):
            count = 68 if split == "valid_seen" else 50
            return [{"id": f"{split}/{i}", "official_split": split,
                     "task_type": "pick_heat_then_place_in_recep", "object": str(i),
                     "target": "target", "scene": str(i)} for i in range(count)]
        config = {"seed": 42, "data_root": "data/alfworld",
                  "splits": {"train": 30, "extension": 58, "test": 20, "allow_reduce": True},
                  "pool_transfer": {"extension_to_train": 10},
                  "evolution": {"rounds": 30, "original_order": "manifest",
                                "extension_without_replacement": True}}
        with tempfile.TemporaryDirectory() as temp, patch("experience.data.collect", catalog):
            before, after = Path(temp) / "before", Path(temp) / "after"
            prepare(config, before)
            old = read_json(before / "data/manifests/development.json")
            updated = deepcopy(config)
            updated["splits"].update(train=40, extension=48)
            updated["pool_transfer"]["extension_to_train"] = 20
            updated["evolution"]["rounds"] = 40
            summary = prepare(updated, after)
            new = read_json(after / "data/manifests/development.json")
            plan = read_json(after / "data/manifests/plan.json")
            self.assertEqual(summary["actual"], {"train": 40, "extension": 48, "test": 20})
            self.assertEqual(new["train"][:30], old["train"])
            added = {task["id"] for task in new["train"][30:]}
            self.assertEqual(len(added), 10)
            self.assertTrue(added <= {task["id"] for task in old["extension"]})
            self.assertEqual({task["id"] for task in new["extension"]},
                             {task["id"] for task in old["extension"]} - added)
            self.assertEqual(read_json(before / "data/manifests/test.json"),
                             read_json(after / "data/manifests/test.json"))
            self.assertEqual([row["original"] for row in plan], new["train"])
            self.assertEqual(len(plan), 40)
            self.assertEqual(len({row["extension"]["id"] for row in plan}), 40)


if __name__ == "__main__":
    unittest.main()
