from .io import dumps
from .environment import task_goal


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}
EXPERIENCE = object_schema({
    "state": TEXT, "explanation": TEXT, "advice": TEXTS, "exceptions": TEXTS,
})
JUDGMENT = object_schema({**EXPERIENCE["properties"], "phase": {"enum": ["planning", "execution"]}})
PLAN = object_schema({"name": TEXT, "strategy": TEXT, "steps": TEXTS})
PLANS = object_schema({"plans": {"type": "array", "items": PLAN, "minItems": 3, "maxItems": 3}})
CRITIQUE = object_schema({
    "trajectory_reviews": {"type": "array", "items": TEXT, "minItems": 3, "maxItems": 3},
    "judge_experiences": {"type": "array", "items": JUDGMENT},
    "planner_experiences": {"type": "array", "items": EXPERIENCE},
    "actor_experiences": {"type": "array", "items": EXPERIENCE},
})


class Roles:
    def __init__(self, client, prompts):
        self.client = client
        self.prompts = {p.stem: p.read_text(encoding="utf-8") for p in prompts.glob("*.txt")}

    def call(self, role, data, schema, seed, purpose):
        return self.client.chat(role, [{"role": "system", "content": self.prompts[role]},
                                       {"role": "user", "content": dumps(data)}], schema, seed, purpose)

    def plan(self, initial, experiences, seed, purpose):
        return self.call("planner", {"task_goal": task_goal(initial), "initial_observation": initial,
                                     "experiences": experiences},
                         PLANS, seed, purpose)["plans"]

    def forecast(self, phase, context, judgments, catalogue, seed, purpose):
        schema = object_schema({"possible_problems": TEXTS,
                                "selected_state_ids": {"type": "array", "uniqueItems": True,
                                  "items": {"enum": [c["state_id"] for c in catalogue]}}})
        return self.call("forecaster", {"phase": phase, "context": context,
                         "judgment_experiences": judgments, "catalogue": catalogue}, schema, seed, purpose)

    def act(self, messages, seed, purpose):
        return self.client.chat("actor", messages, object_schema({"action": TEXT}), seed, purpose)["action"]

    def critique(self, initial, plans, traces, seed, purpose):
        # The compact representation keeps every action and observation; no diagnostic labels are supplied.
        data = {"task_goal": task_goal(initial), "initial_observation": initial, "plans": plans,
                "step_fields": ["action", "observation", "won"],
                "trajectories": [{"plan_index": i, "won": t["won"], "termination": t["termination"],
                                  "steps": [[s["action"], s["observation"], s["won"]] for s in t["steps"]]}
                                 for i, t in enumerate(traces)]}
        return self.call("critic", data, CRITIQUE, seed, purpose)

