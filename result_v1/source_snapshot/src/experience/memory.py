from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

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


def experience_with_id(candidate, candidate_id):
    """Copy model output and attach bookkeeping only; do not filter its content."""
    result = deepcopy(candidate)
    result["id"] = candidate_id
    return result


