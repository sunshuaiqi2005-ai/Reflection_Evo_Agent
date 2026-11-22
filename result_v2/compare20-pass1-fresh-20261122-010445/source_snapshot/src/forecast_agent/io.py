import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16) & 0x7fffffff


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def event(directory, kind, **fields):
    path = Path(directory) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(dumps({"time": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}) + "\n")
