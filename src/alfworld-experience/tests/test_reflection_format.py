from copy import deepcopy
import json
from pathlib import Path
import unittest
import jsonschema

from experience.cognition import CANDIDATE, REVIEW, Cognition, reflection_payload
from experience.memory import Memory
from experience.visible_state import visible_state

ROOT = Path(__file__).resolve().parents[1]


def make_trace(identifier, observation="Your task: 清洗后放置。\nYou see a sinkbasin 1."):
    return {"id": identifier, "task": {"task_type": "pick_clean_then_place_in_recep"},
            "initial_observation": observation, "success": False, "termination": "step_limit",
            "steps": [{"step": index, "action": "clean bowl 1 with sinkbasin 1",
                       "observation": "Nothing happens.\n检查位置。", "score": 0.0, "won": False}
                      for index in range(1, 51)]}


class ReflectionFormatTests(unittest.TestCase):
    def test_proposal_places_task_and_results_first_without_rendering_old_memory(self):
        class Recorder:
            def call(self, messages, seed, purpose, schema):
                self.messages = messages
                return {"captured": True}
        class UnusedMemory:
            def render(self):
                raise AssertionError("Old memory must not be supplied to proposal generation")
        instruction = "put a clean target in the goal receptacle."
        pair = [make_trace(name, "Room description.\nYour task is to: " + instruction) for name in ("a", "b")]
        client = Recorder()
        Cognition(ROOT, client, 42).propose(UnusedMemory(), pair, 1)
        body = client.messages[1]["content"]
        tasks = json.loads(body.split("当前任务原文：\n", 1)[1].splitlines()[0])
        results = json.loads(body.split("两条轨迹的真实结果：\n", 1)[1].splitlines()[0])
        self.assertEqual(tasks, [{"context": 0, "task_type": pair[0]["task"]["task_type"], "instruction": instruction}])
        self.assertEqual(results, [{"id": t["id"], "context": 0, "success": t["success"],
                                    "termination": t["termination"]} for t in pair])
        self.assertLess(body.index("当前任务原文："), body.index("完整执行轨迹："))

    def test_candidate_and_extension_review_require_no_citation_fields(self):
        candidate = {"trigger": "the target is held", "advice": "Visit the appliance before processing.",
                     "exceptions": []}
        jsonschema.validate(candidate, CANDIDATE)
        review = {"trigger": "present", "followed": "no", "contribution": "uncertain",
                  "needs_revision": True, "issue": "Clarify the navigation precondition.", "revision": candidate}
        jsonschema.validate(review, REVIEW)
        self.assertEqual(set(CANDIDATE["properties"]), {"trigger", "advice", "exceptions"})
        for schema in (CANDIDATE, REVIEW):
            self.assertFalse({"evidence", "quote", "trajectory_id", "step"} & set(schema["properties"]))

    def test_full_fifty_step_pair_round_trips_without_losing_feedback_or_results(self):
        pair = [make_trace("a"), make_trace("b")]
        pair[1]["success"] = True
        pair[1]["termination"] = "won"
        pair[1]["steps"][-1].update(score=1.0, won=True)
        original = deepcopy(pair)
        payload = json.loads(json.dumps(reflection_payload(pair), ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(len(payload["contexts"]), 1)
        for source, packed in zip(pair, payload["trajectories"]):
            context = payload["contexts"][packed["context"]]
            self.assertEqual(context["initial_observation"], source["initial_observation"])
            self.assertEqual(context["task_type"], source["task"]["task_type"])
            self.assertEqual(packed["success"], source["success"])
            self.assertEqual(packed["termination"], source["termination"])
            self.assertEqual(packed["final_visible_state"], visible_state(source["steps"]))
            restored = [dict(zip(payload["step_fields"], row)) for row in packed["steps"]]
            self.assertEqual(restored, source["steps"])
        self.assertEqual(pair, original)

    def test_different_original_and_extension_observations_never_share_context(self):
        traces = [make_trace("original"), make_trace("extension", "A different room and task.")]
        packed = reflection_payload(traces)
        self.assertEqual(len(packed["contexts"]), 2)
        self.assertEqual([t["context"] for t in packed["trajectories"]], [0, 1])

    def test_context_sharing_requires_matching_task_type_as_well_as_observation(self):
        traces = [make_trace("a"), make_trace("b")]
        traces[1]["task"]["task_type"] = "pick_cool_then_place_in_recep"
        self.assertEqual(len(reflection_payload(traces)["contexts"]), 2)

    def test_proposal_and_review_both_send_the_lossless_format(self):
        class Recorder:
            def call(self, messages, seed, purpose, schema):
                self.messages = messages
                self.purpose = purpose
                return {"captured": True}
        client = Recorder()
        cognition = Cognition(ROOT, client, 42)
        pair = [make_trace("a"), make_trace("b")]
        cognition.propose(Memory(), pair, 1)
        body = client.messages[1]["content"]
        packed = json.loads(body.strip().splitlines()[-1])
        self.assertEqual(packed, reflection_payload(pair))
        self.assertEqual(len(packed["contexts"]), 1)
        self.assertEqual(packed["contexts"][0]["initial_observation"], pair[0]["initial_observation"])
        candidate = {"advice": "Preserve exact strings: a, b and 中文。"}
        cognition.review(Memory(), candidate, pair, 1)
        body = client.messages[1]["content"]
        packed = json.loads(body.strip().splitlines()[-1])
        self.assertEqual(packed, reflection_payload(pair))
        self.assertIn(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")), body)


if __name__ == "__main__":
    unittest.main()
