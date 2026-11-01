"""Opt-in real-environment checks; controlled commands are never supplied to the actor."""
import json
import os
from pathlib import Path
import unittest

from experience.episode import AlfworldEnvironment, normalize_action

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("ALFWORLD_INTEGRATION") == "1", "Set ALFWORLD_INTEGRATION=1 with the official dataset installed")
class NativeEnvironmentTests(unittest.TestCase):
    def create_ready_environment(self):
        fixture = json.loads((ROOT / "tests/fixtures/placement_regression.json").read_text())
        env = AlfworldEnvironment(ROOT / "data/alfworld/json_2.1.1" / fixture["game_path"], fixture["environment_seed"])
        env.reset()
        self.addCleanup(env.close)
        for action in fixture["prefix_actions"]:
            observation, score, done, won = env.step(action)
            self.assertNotEqual(observation, "Nothing happens.", action)
        return env, fixture

    def test_correct_native_placement_completes_real_task(self):
        env, fixture = self.create_ready_environment()
        obs, _, _, won = env.step(fixture["legacy_placement"])
        self.assertEqual(obs, "Nothing happens.")
        self.assertFalse(won)
        obs, _, done, won = env.step(fixture["native_placement"])
        self.assertTrue(won)
        self.assertTrue(done)

    def test_legacy_alias_completes_real_task_after_normalization(self):
        env, fixture = self.create_ready_environment()
        obs, _, done, won = env.step(normalize_action(fixture["legacy_placement"]))
        self.assertTrue(won)
        self.assertTrue(done)

    def test_global_prompt_uses_the_installed_textworld_placement_dialect(self):
        prompt = (ROOT / "prompts/actor.txt").read_text()
        self.assertIn("move OBJECT to RECEPTACLE", prompt)
        self.assertNotIn("put OBJECT in/on RECEPTACLE", prompt)


if __name__ == "__main__":
    unittest.main()
