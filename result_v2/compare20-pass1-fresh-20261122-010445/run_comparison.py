from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml

directory = Path(__file__).resolve().parent
project = directory.parents[1]
sys.path.insert(0, str(directory / "source_snapshot/src"))

from forecast_agent.client import Ollama
from forecast_agent.experiment import Experiment, report
from forecast_agent.io import read_json, write_json
from forecast_agent.roles import Roles


config = yaml.safe_load((directory / "config.yaml").read_text())
client = Ollama(config, project, directory)
write_json(directory / "metadata.json", client.metadata())
roles = Roles(client, directory / "source_snapshot/prompts")
experiment = Experiment(config, project, directory, roles)

memory = experiment.train(read_json(directory / "training_tasks.json"))
evaluation = experiment.evaluate(read_json(directory / "evaluation_tasks.json"), memory)
primary = {}
for source_name, name in (("initial", "empty_memory"), ("final", "frozen_evolved_memory")):
    metrics = evaluation["metrics"][source_name]
    primary[name] = {
        "questions": metrics["tasks"],
        "trajectories": metrics["trajectories"],
        "successful_trajectories": metrics["successes"],
        "pass_at_1": metrics["success_rate"],
        "mean_steps": metrics["mean_steps"],
    }
write_json(directory / "pass_at_1.json", primary)
report(directory)
info = read_json(directory / "run_info.json")
info["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
write_json(directory / "run_info.json", info)
print("COMPARISON_COMPLETE", primary, flush=True)
