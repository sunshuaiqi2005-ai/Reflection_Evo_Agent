from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re

from .util import ProtocolError, digest


@dataclass
class Memory:
    version: int = 0
    items: list = field(default_factory=list)
    groups: list = field(default_factory=list)

    def to_dict(self):
        return deepcopy({"version": self.version, "items": self.items, "groups": self.groups})

    @classmethod
    def from_dict(cls, data):
        return cls(**deepcopy(data))

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    def appended(self, item):
        result = Memory.from_dict(self.to_dict())
        if item["id"] in {entry["id"] for entry in result.items}:
            raise ProtocolError("Duplicate experience ID")
        result.items.append(deepcopy(item))
        result.groups.append({"source_ids": [item["id"]], "relation": "independent"})
        return result

    def render(self):
        if not self.items:
            return "(No learned experiences.)"
        by_id = {item["id"]: item for item in self.items}
        groups = self.groups or [{"source_ids": [i["id"]], "relation": "independent"}
                                 for i in self.items]
        lines = []
        for index, group in enumerate(groups, 1):
            lines.append(f"Experience group {index} ({group['relation']}):")
            seen = set()
            for source in group["source_ids"]:
                item = by_id[source]
                signature = digest({k: item[k] for k in ("trigger", "advice", "exceptions")})
                if signature in seen:
                    continue
                seen.add(signature)
                lines.append(f"- When {item['trigger']}: {item['advice']}")
                for exception in item["exceptions"]:
                    lines.append(f"  Exception: {exception}")
        return "\n".join(lines)


def validate_evidence(evidence, trajectories):
    if not evidence:
        raise ProtocolError("An empirical claim requires trajectory evidence")
    lookup = {trajectory["id"]: trajectory for trajectory in trajectories}
    for item in evidence:
        trace = lookup.get(item["trajectory_id"])
        if trace is None:
            raise ProtocolError("Evidence references an unavailable trajectory")
        step = item["step"]
        if step == 0:
            text = trace["initial_observation"]
        elif 1 <= step <= len(trace["steps"]):
            entry = trace["steps"][step - 1]
            text = entry["action"] + "\n" + entry["observation"]
        else:
            raise ProtocolError("Evidence step is outside the recorded trajectory")
        quote = " ".join(item["quote"].split())
        if not quote or quote not in " ".join(text.split()):
            raise ProtocolError("Evidence quote does not occur in the recorded step")


def validate_candidate(candidate, trajectories, candidate_id):
    validate_evidence(candidate["evidence"], trajectories)
    prose = " ".join([candidate["trigger"], candidate["advice"], *candidate["exceptions"]])
    if re.search(r"\b[a-z][a-z_-]*\s+\d+\b|FloorPlan\d+|trial_T\d+", prose, re.I):
        raise ProtocolError("Experience contains a task-specific numbered entity or scene")
    for trace in trajectories:
        concrete = trace.get("task", {}).get("object", "")
        if not concrete:
            continue
        words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", concrete).lower().replace("_", " ").split()
        variants = {"".join(words), " ".join(words)}
        if any(re.search(r"\b" + r"\s*".join(map(re.escape, term.split())) + r"(?:s|es)?\b", prose, re.I)
               for term in variants):
            raise ProtocolError("Experience names the task's concrete target type; use target object or another transferable role")
    result = deepcopy(candidate)
    result["id"] = candidate_id
    return result


