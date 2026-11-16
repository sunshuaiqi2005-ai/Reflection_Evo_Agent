from copy import deepcopy
from itertools import islice, zip_longest

import numpy as np

from .io import event


def body(experience):
    return {key: experience[key] for key in ("id", "state", "explanation", "advice", "exceptions")}


def vectors(client, experiences):
    encoded = client.embed([e["state"] for e in experiences])
    return encoded / np.linalg.norm(encoded, axis=1, keepdims=True)


class Memory:
    def __init__(self, data=None):
        self.data = deepcopy(data) if data is not None else {
            "version": 0, "next_experience": 1, "next_state": 1,
            "experiences": [], "clusters": [],
        }

    def select(self, target, context, roles, threshold, seed, purpose, directory, max_experiences=20):
        entries = [e for e in self.data["experiences"] if e["type"] == target]
        phase = {"planner": "planning", "actor": "execution"}[target]
        clusters = [c for c in self.data["clusters"] if (c["type"], c["phase"]) == (target, phase)]
        if len(entries) < threshold:
            result = {"mode": "full", "possible_problems": [], "state_ids": [],
                      "experiences": [body(e) for e in entries]}
        else:
            catalogue = [{k: c[k] for k in ("state_id", "state")}
                         for c in clusters]
            judgments = [body(e) for e in self.data["experiences"]
                         if e["type"] == "judge" and e["phase"] == phase]
            choice = roles.forecast(phase, context, judgments, catalogue, seed, purpose)
            selected = set(choice["selected_state_ids"])
            member_ids = {identifier for c in clusters if c["state_id"] in selected
                          for identifier in c["experience_ids"]}
            result = {"mode": "catalogue", "possible_problems": choice["possible_problems"],
                      "state_ids": choice["selected_state_ids"],
                      "experiences": [body(e) for e in entries if e["id"] in member_ids]}
        result["candidate_count"] = len(result["experiences"])
        result["budget_applied"] = result["candidate_count"] > max_experiences
        if result["budget_applied"]:
            state_order = result["state_ids"] or [c["state_id"] for c in clusters]
            members = {c["state_id"]: c["experience_ids"] for c in clusters}
            groups = [members[state_id] for state_id in state_order]
            ordered_ids = (identifier for row in zip_longest(*groups) for identifier in row
                           if identifier is not None)
            lookup = {e["id"]: e for e in result["experiences"]}
            result["experiences"] = [lookup[i] for i in islice(ordered_ids, max_experiences)]
        event(directory, "injection", purpose=purpose, target=target, memory_version=self.data["version"],
              mode=result["mode"], possible_problems=result["possible_problems"],
              candidate_count=result["candidate_count"], max_experiences=max_experiences,
              budget_applied=result["budget_applied"],
              state_ids=result["state_ids"], experience_ids=[e["id"] for e in result["experiences"]])
        return result

    def add(self, experience_type, phase, content):
        entry = {**content, "id": f"E{self.data['next_experience']:05d}",
                 "type": experience_type, "phase": phase}
        self.data["next_experience"] += 1
        self.data["experiences"].append(entry)
        return entry

    def updated(self, critique, client, similarity_threshold):
        result = Memory(self.data)
        pending = []
        for key, kind in (("judge_experiences", "judge"), ("planner_experiences", "planner"),
                          ("actor_experiences", "actor")):
            for content in critique[key]:
                phase = content["phase"] if kind == "judge" else {"planner": "planning", "actor": "execution"}[kind]
                value = {k: v for k, v in content.items() if k != "phase"}
                if any(e["type"] == kind and e["phase"] == phase
                       and {k: e[k] for k in value} == value
                       for e in result.data["experiences"]):
                    continue
                pending.append(result.add(kind, phase, value))
        if not pending:
            result.data["version"] += 1
            return result
        entries = result.data["experiences"]
        lookup = {e["id"]: e for e in entries}
        encoded = {e["id"]: v for e, v in zip(entries, vectors(client, entries))}
        for entry in pending:
            vector = encoded[entry["id"]]
            candidates = [c for c in result.data["clusters"]
                          if (c["type"], c["phase"]) == (entry["type"], entry["phase"])]
            nearest = max(candidates, key=lambda c: np.dot(vector, c["centroid"]), default=None)
            if nearest is None or np.dot(vector, nearest["centroid"]) < similarity_threshold:
                nearest = {"state_id": f"S{result.data['next_state']:05d}", "type": entry["type"],
                           "phase": entry["phase"], "state": entry["state"],
                           "experience_ids": [], "centroid": vector.tolist()}
                result.data["next_state"] += 1
                result.data["clusters"].append(nearest)
            nearest["experience_ids"].append(entry["id"])
            entry["state_id"] = nearest["state_id"]
            members = np.asarray([encoded[i] for i in nearest["experience_ids"]])
            center = members.mean(axis=0)
            center /= np.linalg.norm(center)
            nearest["centroid"] = center.tolist()
            representative = nearest["experience_ids"][int(np.argmax(members @ center))]
            nearest["state"] = lookup[representative]["state"]
        result.data["version"] += 1
        return result
