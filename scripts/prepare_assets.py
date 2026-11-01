"""Fetch official text-only ALFWorld assets; no visual model or expert agent checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "json_2.1.1_json.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "json_2.1.1_pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "json_2.1.3_tw-pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
}


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    subprocess.run(["/usr/bin/curl", "-fL", "--retry", "3", "--connect-timeout", "20",
                    "--max-time", "1200", "-C", "-", url, "-o", str(partial)], check=True)
    partial.replace(destination)


def main():
    output = ROOT / "data/alfworld"
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, url in ASSETS.items():
        archive = ROOT / "downloads" / name
        download(url, archive)
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if not (output / entry.filename).resolve().is_relative_to(output):
                    raise ValueError("Unsafe archive path")
            if zipped.testzip():
                raise ValueError("Damaged ZIP: " + name)
            for entry in zipped.infolist():
                target = output / entry.filename
                if not target.exists() or target.stat().st_size != entry.file_size:
                    zipped.extract(entry, output)
        sources.append({"file": name, "url": url, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    os.environ["ALFWORLD_DATA"] = str(output)
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    logic = output / "logic"
    logic.mkdir(exist_ok=True)
    shutil.copy2(ALFRED_PDDL_PATH, logic / "alfred.pddl")
    shutil.copy2(ALFRED_TWL2_PATH, logic / "alfred.twl2")
    tokenizer = ROOT / "data/tokenizer/tokenizer.json"
    tokenizer_url = "https://modelscope.cn/models/Qwen/Qwen3-14B/resolve/master/tokenizer.json"
    download(tokenizer_url, tokenizer)
    from tokenizers import Tokenizer
    Tokenizer.from_file(str(tokenizer))
    sources.append({"file": "tokenizer.json", "url": tokenizer_url,
                    "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()})
    (ROOT / "downloads/data-sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print("Official text-only assets are ready.")


if __name__ == "__main__":
    main()
