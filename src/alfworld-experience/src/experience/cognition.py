from __future__ import annotations

import json

from .util import seed_for
from .visible_state import visible_state


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": required or list(properties), "additionalProperties": False}


def text(maximum=500):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


CANDIDATE = object_schema({"trigger": text(160), "advice": text(360),
                           "exceptions": {"type": "array", "maxItems": 3, "items": text(160)}})
PROPOSAL = object_schema({"analysis": text(1400), "candidate": CANDIDATE})
REVIEW = object_schema({
    "trigger": {"enum": ["present", "absent", "uncertain"]},
    "followed": {"enum": ["yes", "no", "uncertain"]},
    "contribution": {"enum": ["helped", "harmed", "uncertain", "not_applicable"]},
    "needs_revision": {"type": "boolean"},
    "issue": {"type": "string", "maxLength": 600},
    "revision": {"anyOf": [CANDIDATE, {"type": "null"}]},
})


STEP_FIELDS = ("step", "action", "observation", "score", "won")


def reflection_payload(traces):
    """Lossless reflection view: share identical context, declare step keys once."""
    contexts, trajectories = [], []
    for trace in traces:
        context = {"task_type": trace["task"]["task_type"],
                   "initial_observation": trace["initial_observation"]}
        if context not in contexts:
            contexts.append(context)
        trajectories.append({"id": trace["id"], "context": contexts.index(context),
                             "success": trace["success"], "termination": trace["termination"],
                             "final_visible_state": visible_state(trace["steps"]),
                             "steps": [[step[key] for key in STEP_FIELDS] for step in trace["steps"]]})
    return {"step_fields": list(STEP_FIELDS), "contexts": contexts, "trajectories": trajectories}


class Cognition:
    def __init__(self, project, client, seed):
        self.project, self.client, self.seed = project, client, seed

    def ask(self, name, schema, round_index, **values):
        template = (self.project / f"prompts/{name}.txt").read_text()
        filled = template.format(**{key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                    for key, value in values.items()})
        messages = [{"role": "system", "content": "Analyze the supplied execution trajectories. Return only the requested JSON."},
                    {"role": "user", "content": filled}]
        return self.client.call(messages, seed_for(self.seed, "cognition", round_index, name),
                                f"round-{round_index:02d}/{name}", schema)

    def propose(self, memory, pair, round_index):
        return self.ask("propose", PROPOSAL, round_index, memory=memory.render(),
                        trajectories=reflection_payload(pair))

    def review(self, memory, candidate, traces, round_index):
        return self.ask("review", REVIEW, round_index, memory=memory.render(), candidate=candidate,
                        trajectories=reflection_payload(traces))

