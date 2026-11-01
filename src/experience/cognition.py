from __future__ import annotations

import json

from .util import seed_for


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": required or list(properties), "additionalProperties": False}


def text(maximum=500):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


EVIDENCE = {"type": "array", "minItems": 1, "maxItems": 4, "items": object_schema({
    "trajectory_id": text(160), "step": {"type": "integer", "minimum": 0}, "quote": text(160)})}
CANDIDATE = object_schema({"trigger": text(160), "advice": text(360),
                           "exceptions": {"type": "array", "maxItems": 3, "items": text(160)},
                           "evidence": EVIDENCE})
PROPOSAL = object_schema({"analysis": text(1400), "candidate": CANDIDATE})
REVIEW = object_schema({
    "trigger": {"enum": ["present", "absent", "uncertain"]},
    "followed": {"enum": ["yes", "no", "uncertain"]},
    "contribution": {"enum": ["helped", "harmed", "uncertain", "not_applicable"]},
    "needs_revision": {"type": "boolean"},
    "issue": {"type": "string", "maxLength": 600},
    "evidence": {"type": "array", "maxItems": 4, "items": EVIDENCE["items"]},
    "revision": {"anyOf": [CANDIDATE, {"type": "null"}]},
})


def reflection_trace(trace):
    return {"id": trace["id"], "task_type": trace["task"]["task_type"],
            "initial_observation": trace["initial_observation"], "success": trace["success"],
            "termination": trace["termination"],
            "steps": [{key: step[key] for key in ("step", "action", "observation", "score", "won")}
                      for step in trace["steps"]]}


class Cognition:
    def __init__(self, project, client, seed):
        self.project, self.client, self.seed = project, client, seed

    def ask(self, name, schema, round_index, **values):
        template = (self.project / f"prompts/{name}.txt").read_text()
        filled = template.format(**{key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                                    for key, value in values.items()})
        messages = [{"role": "system", "content": "You are an evidence-grounded research assistant. Return only the requested JSON."},
                    {"role": "user", "content": filled}]
        return self.client.call(messages, seed_for(self.seed, "cognition", round_index, name),
                                f"round-{round_index:02d}/{name}", schema)

    def propose(self, memory, pair, round_index):
        return self.ask("propose", PROPOSAL, round_index, memory=memory.render(),
                        trajectories=[reflection_trace(t) for t in pair])

    def review(self, memory, candidate, traces, round_index):
        return self.ask("review", REVIEW, round_index, memory=memory.render(), candidate=candidate,
                        trajectories=[reflection_trace(t) for t in traces])

