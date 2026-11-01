from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

from .util import digest, read_json, seed_for, write_json

TASK_TYPES = {
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
}


def collect(root, split):
    tasks = []
    for path in sorted((Path(root) / split).rglob("game.tw-pddl")):
        relative = path.relative_to(root).as_posix()
        if "movable" in relative or "Sliced" in relative:
            continue
        game = read_json(path)
        if game.get("solvable") is not True or not (path.parent / "traj_data.json").is_file():
            continue
        metadata = read_json(path.parent / "traj_data.json")
        if metadata["task_type"] not in TASK_TYPES:
            continue
        params = metadata.get("pddl_params", {})
        tasks.append({"id": relative, "game_path": relative, "official_split": split,
                      "task_type": metadata["task_type"],
                      "object": params.get("object_target", ""),
                      "target": params.get("parent_target", ""),
                      "scene": str(metadata.get("scene", {}).get("scene_num", "")),
                      "game_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return tasks


def stratified_sample(tasks, count, seed):
    """Round-robin across task families; shuffle both buckets and family order."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for task in sorted(tasks, key=lambda t: t["id"]):
        buckets[task["task_type"]].append(task)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    families = sorted(buckets)
    rng.shuffle(families)
    chosen = []
    while len(chosen) < count and any(buckets.values()):
        for family in families:
            if buckets[family] and len(chosen) < count:
                chosen.append(buckets[family].pop())
    rng.shuffle(chosen)
    return chosen


def random_sample(tasks, count, seed):
    """Uniform sampling without replacement; no family balancing of training tasks."""
    ordered = sorted(tasks, key=lambda task: task["id"])
    return random.Random(seed).sample(ordered, min(count, len(ordered)))


def choose_extension(original, pool, seed):
    same_family = [t for t in pool if t["task_type"] == original["task_type"]]
    if not same_family:
        raise ValueError("扩展池没有同任务结构题目，不能悄悄用不匹配题替代。")
    def variety(task):
        return sum(task[key] != original[key] for key in ("object", "target", "scene"))
    best = max(map(variety, same_family))
    choices = sorted([t for t in same_family if variety(t) == best], key=lambda t: t["id"])
    return random.Random(seed).choice(choices)


def prepare(config, project):
    root = project / config["data_root"]
    selection = config.get("task_selection", {})
    training_types = selection.get("train_task_types")
    if training_types is not None and (not training_types or set(training_types) - TASK_TYPES):
        raise ValueError("train_task_types must contain supported ALFWorld task types")
    output = project / "data/manifests"
    signature = digest({"seed": config["seed"], "splits": config["splits"],
                        "evolution": config["evolution"], "root": str(root),
                        "task_selection": selection,
                        "train_sampling": "uniform_without_replacement_v2"})
    existing = output / "summary.json"
    if existing.exists():
        summary = read_json(existing)
        if summary["configuration_hash"] != signature:
            raise ValueError("题池配置已变化。请将 data/manifests 另行备份后重新 prepare，避免覆盖既有划分。")
        for name, expected in summary["manifest_hashes"].items():
            if digest(read_json(output / name)) != expected:
                raise ValueError("题目清单已被修改：" + name)
        return summary
    pools, available, warnings = {}, {}, []
    for name, official in (("train", "train"), ("extension", "valid_seen"), ("test", "valid_unseen")):
        tasks = collect(root, official)
        if name == "train" and training_types is not None:
            tasks = [task for task in tasks if task["task_type"] in training_types]
        available[name] = len(tasks)
        requested = config["splits"][name]
        if len(tasks) < requested:
            if not config["splits"]["allow_reduce"]:
                raise ValueError(f"{name}: requested {requested}, available {len(tasks)}")
            warnings.append(f"{name} reduced from {requested} to {len(tasks)}")
        sampler = random_sample if name == "train" else stratified_sample
        pools[name] = sampler(tasks, min(requested, len(tasks)),
                              seed_for(config["seed"], "split", name))
    identifiers = [t["id"] for pool in pools.values() for t in pool]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("发现题池交集")
    requested_rounds = config["evolution"]["rounds"]
    rounds = min(requested_rounds, len(pools["train"]), len(pools["extension"]))
    if rounds < requested_rounds:
        warnings.append(f"rounds reduced from {requested_rounds} to {rounds}")
    if rounds == 0 or not pools["test"]:
        raise ValueError("题目不足，无法建立实验")
    originals = random_sample(pools["train"], rounds,
                              seed_for(config["seed"], "round-originals"))
    remaining = list(pools["extension"])
    plan = []
    for index, original in enumerate(originals, 1):
        extension_seed = seed_for(config["seed"], "extension", index)
        extension = choose_extension(original, remaining, extension_seed)
        if config["evolution"]["extension_without_replacement"]:
            remaining = [t for t in remaining if t["id"] != extension["id"]]
        plan.append({"round": index, "original": original, "extension": extension,
                     "extension_seed": extension_seed})
    manifests = {"development.json": {"train": pools["train"], "extension": pools["extension"]},
                 "test.json": {"test": pools["test"]}, "plan.json": plan}
    for name, value in manifests.items():
        write_json(output / name, value)
    summary = {"configuration_hash": signature, "seed": config["seed"],
               "task_selection": selection,
               "available": available, "requested": config["splits"],
               "actual": {k: len(v) for k, v in pools.items()}, "rounds": rounds,
               "sampling": {"train_pool": "uniform_without_replacement",
                            "round_originals": "uniform_without_replacement",
                            "extension_pool": "stratified", "test_pool": "stratified"},
               "family_counts": {k: dict(Counter(t["task_type"] for t in v)) for k, v in pools.items()},
               "manifest_hashes": {name: digest(value) for name, value in manifests.items()},
               "warnings": warnings}
    write_json(existing, summary)
    return summary
