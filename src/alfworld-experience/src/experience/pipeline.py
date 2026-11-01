from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random

from .memory import Memory, validate_candidate, validate_evidence
from .util import ProtocolError, digest, event, read_json, seed_for, write_json

PROTOCOL = "original_once_extension_once_append"

def choose_pair(traces, seed):
    successes = [t for t in traces if t["success"]]
    failures = [t for t in traces if not t["success"]]
    rng = random.Random(seed)
    if not failures:
        return None, "all_success"
    if successes:
        return [rng.choice(successes), rng.choice(failures)], "success_failure"
    def behavior(trace):
        return digest([(step["action"], step["observation"]) for step in trace["steps"]])
    pairs = list(combinations(failures, 2))
    different = [pair for pair in pairs if behavior(pair[0]) != behavior(pair[1])]
    return list(rng.choice(different or pairs)), "failure_failure"


class Evolution:
    def __init__(self, config, run_dir, runner, cognition):
        self.config, self.directory, self.runner, self.cognition = config, Path(run_dir), runner, cognition
        if config["evolution"]["trials_per_task"] != 3 or config["evolution"]["max_revisions"] != 1:
            raise ValueError("第一版协议固定为三次独立运行、最多一次修订。")

    def round(self, plan, memory):
        index = plan["round"]
        folder = self.directory / "rounds" / f"{index:02d}"
        path = folder / "result.json"
        before = memory.fingerprint
        if path.exists():
            cached = read_json(path)
            if (cached.get("protocol") != PROTOCOL or
                    cached["memory_before_hash"] != before or cached["plan"] != plan):
                raise RuntimeError("Round resume inputs do not match")
            restored = Memory.from_dict(cached["memory_after"])
            if restored.fingerprint != cached["memory_after_hash"]:
                raise RuntimeError("Saved round memory checksum does not match")
            return restored, cached
        folder.mkdir(parents=True, exist_ok=True)
        result = {"protocol": PROTOCOL, "round": index, "plan": plan, "memory_before_hash": before,
                  "memory_before": memory.to_dict(), "episodes": {}, "status": None}
        event(self.directory, "round_start", round=index, memory_hash=before)

        def episode(stage, current_memory, task, trial=None):
            suffix = stage + (f"-{trial}" if trial is not None else "")
            actor_seed = seed_for(self.config["seed"], "round", index, suffix)
            identifier = f"r{index:02d}-{suffix}"
            trace = self.runner.run(task, current_memory, actor_seed, identifier, stage)
            result["episodes"][suffix] = {"id": trace["id"], "success": trace["success"],
                                          "steps": len(trace["steps"]), "seed": actor_seed,
                                          "memory_hash": current_memory.fingerprint}
            return trace

        next_memory = memory
        try:
            # No proposal is generated until all three independent baseline episodes finish.
            trials = [episode("trial", memory, plan["original"], i) for i in range(1, 4)]
            if memory.fingerprint != before:
                raise RuntimeError("Frozen M_t was mutated during baseline trials")
            pairing_seed = seed_for(self.config["seed"], "pair", index)
            pair, pair_type = choose_pair(trials, pairing_seed)
            result["pairing"] = {"seed": pairing_seed, "type": pair_type,
                                  "trajectory_ids": [t["id"] for t in pair] if pair else []}
            if pair is None:
                result["status"] = "skipped_all_success"
            else:
                proposal = self.cognition.propose(memory, pair, index)
                write_json(folder / "proposal.json", proposal)
                candidate = validate_candidate(proposal["candidate"], pair, f"r{index:02d}-candidate")
                result["candidate"] = candidate
                temporary = memory.appended(candidate)
                original = episode("candidate-original", temporary, plan["original"])
                if not original["success"]:
                    result["status"] = "rejected_original"
                else:
                    extension = episode("candidate-extension", temporary, plan["extension"])
                    review = self.cognition.review(memory, candidate, [original, extension], index)
                    write_json(folder / "extension_review.json", review)
                    if review["evidence"]:
                        validate_evidence(review["evidence"], [original, extension])
                    revised = review["needs_revision"]
                    if review["needs_revision"]:
                        if (not review["issue"].strip() or not review["evidence"] or
                                review["revision"] is None):
                            raise ProtocolError("Revision lacks a specific, evidence-supported correction")
                        candidate = validate_candidate(review["revision"], [original, extension], f"r{index:02d}-revised")
                        if not any(e["trajectory_id"] == extension["id"]
                                   for e in [*review["evidence"], *candidate["evidence"]]):
                            raise ProtocolError("Extension-based revision must cite the extension trajectory")
                        result["revised_candidate"] = candidate
                    elif review["revision"] is not None:
                        raise ProtocolError("Revision supplied while needs_revision is false")
                    # The extension supplies correction evidence, not an acceptance success gate.
                    # Commit immediately after this one review; never run revised/regression episodes.
                    candidate["validated_by"] = [] if revised else [original["id"]]
                    candidate["reviewed_on"] = [extension["id"]]
                    candidate["acceptance"] = {
                        "basis": "original_pass_then_extension_review",
                        "original_validation_episode": original["id"],
                        "originally_validated_candidate": result["candidate"]["id"],
                        "extension_success": extension["success"],
                        "revised": revised,
                        "post_review_revalidated": False,
                    }
                    next_memory = memory.appended(candidate)
                    next_memory.version = memory.version + 1
                    result["accepted_experience"] = candidate
                    result["status"] = "accepted"
                    write_json(folder / "accepted_experience.json", candidate)
        except ProtocolError as exc:
            result["status"] = "rejected_protocol"
            result["protocol_error"] = str(exc)
        # Infrastructure exceptions propagate: a broken API/environment must never become a failure label.
        if memory.fingerprint != before:
            raise RuntimeError("M_t was modified before the commit point")
        result["memory_after"] = next_memory.to_dict()
        result["memory_after_hash"] = next_memory.fingerprint
        write_json(path, result)
        event(self.directory, "round_end", round=index, status=result["status"],
              memory_version=next_memory.version, memory_hash=next_memory.fingerprint)
        return next_memory, result

    def evolve(self, plan):
        initial_path = self.directory / "memory_initial.json"
        initial = Memory.from_dict(read_json(initial_path)) if initial_path.exists() else Memory()
        write_json(initial_path, initial.to_dict())
        memory, results = initial, []
        for entry in plan:
            memory, result = self.round(entry, memory)
            results.append(result)
            write_json(self.directory / "memory_current.json", memory.to_dict())
            write_json(self.directory / "status.json", {"phase": "evolution", "rounds_completed": len(results),
                       "rounds_total": len(plan), "accepted": sum(r["status"] == "accepted" for r in results),
                       "memory_hash": memory.fingerprint})
            print(f"Round {entry['round']}/{len(plan)}: {result['status']}; memory version {memory.version}", flush=True)
        frozen = {"frozen": True, "memory": memory.to_dict(), "memory_hash": memory.fingerprint,
                  "rounds_completed": len(results)}
        write_json(self.directory / "memory_final.json", frozen)
        write_json(self.directory / "status.json", {"phase": "evolution_complete", "rounds_completed": len(results),
                   "accepted": sum(r["status"] == "accepted" for r in results), "memory_hash": memory.fingerprint})
        return frozen


def evaluate(config, directory, tasks, runner):
    directory = Path(directory)
    frozen = read_json(directory / "memory_final.json")
    memory = Memory.from_dict(frozen["memory"])
    if not frozen.get("frozen") or memory.fingerprint != frozen["memory_hash"]:
        raise ValueError("最终测试要求已冻结且未被修改的经验库")
    arms = [("final", memory)]
    if config["evaluation"]["compare_initial_memory"]:
        arms.insert(0, ("initial", Memory.from_dict(read_json(directory / "memory_initial.json"))))
    outcomes = {name: [] for name, _ in arms}
    for index, task in enumerate(tasks, 1):
        # Common sampling seeds across arms; environment resets and histories remain independent.
        actor_seed = seed_for(config["seed"], "heldout-test", task["id"])
        order = list(arms)
        random.Random(seed_for(config["seed"], "test-order", task["id"])).shuffle(order)
        for arm, current in order:
            expected_hash = current.fingerprint
            trace = runner.run(task, current, actor_seed, f"test-{index:03d}-{arm}", "heldout-test")
            if current.fingerprint != expected_hash:
                raise RuntimeError("A test episode modified its frozen experience input")
            outcomes[arm].append({"task_id": task["id"], "success": trace["success"],
                                  "steps": len(trace["steps"]), "episode_id": trace["id"]})
        if read_json(directory / "memory_final.json") != frozen:
            raise RuntimeError("Frozen memory was modified during final testing")
        write_json(directory / "test_progress.json", {"completed": index, "total": len(tasks), "outcomes": outcomes})
        write_json(directory / "status.json", {"phase": "test", "completed": index, "total": len(tasks)})
        print(f"Held-out test {index}/{len(tasks)} complete", flush=True)
    result = {"frozen_memory_hash": frozen["memory_hash"], "tasks": len(tasks), "outcomes": outcomes,
              "metrics": {name: {"successes": sum(t["success"] for t in rows),
                                  "success_rate": sum(t["success"] for t in rows) / len(rows),
                                  "mean_steps": sum(t["steps"] for t in rows) / len(rows)}
                          for name, rows in outcomes.items()}}
    if "initial" in outcomes:
        result["success_rate_delta"] = result["metrics"]["final"]["success_rate"] - result["metrics"]["initial"]["success_rate"]
    write_json(directory / "test_results.json", result)
    write_json(directory / "status.json", {"phase": "complete", "test_metrics": result["metrics"]})
    return result
