from copy import deepcopy

import numpy as np

from .io import event, seed_for


def body(experience):
    return {key: experience[key] for key in ("id", "context", "problem", "explanation",
                                            "methods", "advice", "exceptions")}


def vectors(client, experiences):
    contexts = client.embed([e["context"] for e in experiences])
    problems = client.embed([e["problem"] for e in experiences])
    contexts = contexts / np.linalg.norm(contexts, axis=1, keepdims=True)
    problems = problems / np.linalg.norm(problems, axis=1, keepdims=True)
    return np.concatenate([contexts, problems], axis=1) / np.sqrt(2)


class Memory:
    def __init__(self, data=None):
        self.data = deepcopy(data) if data is not None else {
            "version": 0, "next_experience": 1, "next_cluster": 1,
            "experiences": [], "clusters": [],
        }

    def select(self, target, context, roles, threshold, seed, purpose, directory):
        entries = [e for e in self.data["experiences"] if e["type"] == target]
        phase = {"planner": "planning", "actor": "execution"}[target]
        if len(entries) < threshold:
            result = {"mode": "full", "possible_problems": [], "project_ids": [],
                      "experiences": [body(e) for e in entries]}
        else:
            catalogue = [{k: c[k] for k in ("id", "state_name", "problem_name", "scope")}
                         for c in self.data["clusters"] if c["type"] == target]
            judgments = [body(e) for e in self.data["experiences"]
                         if e["type"] == "judge" and e["phase"] == phase]
            choice = roles.forecast(phase, context, judgments, catalogue, seed, purpose)
            selected = set(choice["selected_project_ids"])
            member_ids = {identifier for c in self.data["clusters"] if c["id"] in selected
                          for identifier in c["experience_ids"]}
            result = {"mode": "catalogue", "possible_problems": choice["possible_problems"],
                      "project_ids": choice["selected_project_ids"],
                      "experiences": [body(e) for e in entries if e["id"] in member_ids]}
        event(directory, "injection", purpose=purpose, target=target, memory_version=self.data["version"],
              mode=result["mode"], possible_problems=result["possible_problems"],
              project_ids=result["project_ids"], experience_ids=[e["id"] for e in result["experiences"]])
        return result

    def add(self, experience_type, phase, content):
        entry = {**content, "id": f"E{self.data['next_experience']:05d}",
                 "type": experience_type, "phase": phase}
        self.data["next_experience"] += 1
        self.data["experiences"].append(entry)
        return entry

    def updated(self, critique, client, roles, similarity_threshold, seed, round_index):
        result = Memory(self.data)
        pending = []
        for key, kind in (("judge_experiences", "judge"), ("planner_experiences", "planner"),
                          ("actor_experiences", "actor")):
            for content in critique[key]:
                phase = content["phase"] if kind == "judge" else {"planner": "planning", "actor": "execution"}[kind]
                pending.append(result.add(kind, phase, {k: v for k, v in content.items() if k != "phase"}))
        if not pending:
            result.data["version"] += 1
            return result
        encoded = vectors(client, pending)
        touched = set()
        for entry, vector in zip(pending, encoded):
            candidates = [c for c in result.data["clusters"]
                          if (c["type"], c["phase"]) == (entry["type"], entry["phase"])]
            nearest = max(candidates, key=lambda c: np.dot(vector, c["centroid"]), default=None)
            if nearest is None or np.dot(vector, nearest["centroid"]) < similarity_threshold:
                nearest = {"id": f"C{result.data['next_cluster']:05d}", "type": entry["type"],
                           "phase": entry["phase"], "state_name": entry["context"],
                           "problem_name": entry["problem"], "scope": entry["context"],
                           "experience_ids": [], "centroid": vector.tolist()}
                result.data["next_cluster"] += 1
                result.data["clusters"].append(nearest)
            size = len(nearest["experience_ids"])
            center = (np.asarray(nearest["centroid"]) * size + vector) / (size + 1)
            nearest["centroid"] = (center / np.linalg.norm(center)).tolist()
            nearest["experience_ids"].append(entry["id"])
            entry["cluster_id"] = nearest["id"]
            touched.add(nearest["id"])

        changed = [c for c in result.data["clusters"] if c["id"] in touched]
        lookup = {e["id"]: e for e in result.data["experiences"]}
        groups = [{"id": c["id"], "type": c["type"], "phase": c["phase"],
                   "experiences": [body(lookup[i]) for i in c["experience_ids"]]} for c in changed]
        merged = roles.consolidate(groups, seed_for(seed, "consolidate", round_index),
                                   f"round-{round_index:03d}/consolidate")
        for cluster in changed:
            combined = merged[cluster["id"]]
            old_ids = set(cluster["experience_ids"])
            result.data["experiences"] = [e for e in result.data["experiences"] if e["id"] not in old_ids]
            entries = [result.add(cluster["type"], cluster["phase"], content)
                       for content in combined["experiences"]]
            for entry in entries:
                entry["cluster_id"] = cluster["id"]
            cluster["experience_ids"] = [e["id"] for e in entries]
            cluster.update({k: combined[k] for k in ("state_name", "problem_name", "scope")})
            center = vectors(client, entries).mean(axis=0)
            cluster["centroid"] = (center / np.linalg.norm(center)).tolist()
        result.data["version"] += 1
        return result
