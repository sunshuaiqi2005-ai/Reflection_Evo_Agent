from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experience.data import prepare, stratified_sample
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
                "exceptions": [], "evidence": [{"trajectory_id": "a", "step": 0, "quote": "A cup is visible."}]}
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


if __name__ == "__main__":
    unittest.main()
