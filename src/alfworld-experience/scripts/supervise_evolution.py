"""Run the authorized ten-round evolution and record progress every 30 minutes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo("Asia/Shanghai")


def timestamp():
    return datetime.now(ZONE).isoformat(timespec="seconds")


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def tail(path, limit=8000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def snapshot(directory, phase, started, worker, interval, **extra):
    rounds = [load(p) for p in sorted((directory / "rounds").glob("*/result.json"))]
    progress = list((directory / "episodes").glob("*.progress.json"))
    active = load(max(progress, key=lambda p: p.stat().st_mtime)) if progress else None
    status = load(directory / "status.json", {})
    result = {"checked_at": timestamp(), "started_at": started,
              "phase": phase, "supervisor_pid": os.getpid(), "worker_pid": worker.pid,
              "check_interval_seconds": interval, "rounds_completed": len(rounds),
              "rounds_total": 10, "accepted_rounds": sum(r["status"] == "accepted" for r in rounds),
              "round_status_counts": dict(Counter(r["status"] for r in rounds)),
              "evolution_status": status, **extra}
    if active:
        result["active_episode"] = {"id": active["id"], "stage": active["stage"],
                                    "completed_steps": len(active["steps"])}
    events = directory / "events.jsonl"
    if events.exists():
        result["seconds_since_last_event"] = round(time.time() - events.stat().st_mtime)
    save(directory / "supervision.json", result)
    with (directory / "monitoring.jsonl").open("a") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def summarize(directory, phase, started, returncode):
    results = [load(p) for p in sorted((directory / "rounds").glob("*/result.json"))]
    memory = load(directory / "memory_final.json")
    rows = []
    for result in results:
        episodes = result["episodes"]
        original = episodes.get("candidate-original")
        extension = episodes.get("candidate-extension")
        rows.append({"round": result["round"], "status": result["status"],
                     "task_type": result["plan"]["original"]["task_type"],
                     "baseline_successes": sum(e["success"] for k, e in episodes.items() if k.startswith("trial-")),
                     "original_validation_success": original["success"] if original else None,
                     "extension_success": extension["success"] if extension else None,
                     "protocol_error": result.get("protocol_error")})
    summary = {"phase": phase, "started_at": started, "finished_at": timestamp(),
               "returncode": returncode, "rounds_completed": len(results),
               "accepted_rounds": sum(r["status"] == "accepted" for r in results),
               "experience_count": len(memory["memory"]["items"]) if memory else None,
               "rounds": rows, "final_test_executed": False}
    save(directory / "evolution_summary.json", summary)
    labels = {"accepted": "接收经验", "rejected_original": "原题验证失败，舍弃",
              "skipped_all_success": "三次均成功，跳过", "rejected_protocol": "经验输出不符合格式或通用性要求"}
    lines = ["# 十轮经验进化结果", "", f"状态：{phase}；完成 {len(results)}/10 轮。",
             f"开始：{started}；结束：{summary['finished_at']}。",
             f"接收经验轮数：{summary['accepted_rounds']}；最终经验条数：{summary['experience_count']}。", "",
             "| 轮次 | 三次初始执行成功数 | 处理 |", "| --- | --- | --- |"]
    lines += [f"| {r['round']} | {r['baseline_successes']}/3 | {labels.get(r['status'], r['status'])} |" for r in rows]
    errors = [r for r in rows if r["protocol_error"]]
    if errors:
        lines += ["", "协议拒绝原因："] + [f"- 第 {r['round']} 轮：{r['protocol_error']}" for r in errors]
    policy = ("扩展题用于修正经验，修正后直接追加，未做复验。"
              if any(r.get("protocol") == "original_once_extension_once_append" for r in results) else
              "扩展题成功直接追加候选；失败才审阅并最多修正一次，随后追加，不再复验。")
    lines += ["", "本次仅执行经验进化，未运行最终测试集；以上不是测试成功率。", policy, ""]
    (directory / "evolution_summary.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/main")
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    directory = (ROOT / args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "memory_final.json").exists():
            raise RuntimeError("This run already has a frozen final memory; refusing duplicate execution")
        started = timestamp()
        worker = None

        def stop(_signal, _frame):
            if worker is not None and worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        command = ["/usr/bin/caffeinate", "-i", str(ROOT / "scripts/run.sh"),
                   "evolve", "--run-dir", str(directory)]
        with (directory / "console.log").open("ab", buffering=0) as output:
            try:
                for attempt in range(3):
                    worker = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                              stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    snapshot(directory, "running", started, worker, args.interval, attempt=attempt + 1)
                    while True:
                        try:
                            code = worker.wait(timeout=args.interval)
                            break
                        except subprocess.TimeoutExpired:
                            snapshot(directory, "running", started, worker, args.interval, attempt=attempt + 1)
                    status = load(directory / "status.json", {})
                    frozen = load(directory / "memory_final.json", {})
                    if code == 0 and status.get("phase") == "evolution_complete" and frozen.get("rounds_completed") == 10:
                        snapshot(directory, "complete", started, worker, args.interval, returncode=code)
                        summarize(directory, "complete", started, code)
                        return
                    recent = tail(directory / "console.log")
                    if "InfrastructureError" in recent and attempt < 2:
                        snapshot(directory, "retrying_infrastructure", started, worker, args.interval,
                                 returncode=code, attempt=attempt + 1)
                        time.sleep(30)
                        startup = ROOT / "scripts/start_ollama.sh"
                        if not startup.exists():
                            startup = ROOT.parent / "qwen-local/启动服务.command"
                        subprocess.run([str(startup)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=60)
                        continue
                    snapshot(directory, "stopped_error", started, worker, args.interval,
                             returncode=code, error_tail=recent)
                    summarize(directory, "stopped_error", started, code)
                    return
            except KeyboardInterrupt:
                snapshot(directory, "interrupted", started, worker, args.interval)
                summarize(directory, "interrupted", started, 130)
                raise


if __name__ == "__main__":
    main()
