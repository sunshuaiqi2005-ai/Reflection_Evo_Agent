from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16) & 0x7FFFFFFF


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def event(directory, kind, **fields):
    path = Path(directory) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def exclusive_run(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("此运行目录已有实验进程，不能同时启动两次。")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class ProtocolError(ValueError):
    """Unusable candidate/fusion/output, never a task-success signal."""


class InfrastructureError(RuntimeError):
    """Stop and resume later; do not count this as an agent failure."""


class ContextBudgetError(ProtocolError):
    pass
