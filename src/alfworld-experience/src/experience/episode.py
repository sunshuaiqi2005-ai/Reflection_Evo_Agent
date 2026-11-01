from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

from .util import ContextBudgetError, InfrastructureError, digest, event, read_json, seed_for, write_json


class AlfworldEnvironment:
    """Official PDDL engine and demangler; no expert or hidden-state input to the actor."""
    def __init__(self, gamefile, seed):
        os.environ["ALFWORLD_DATA"] = str(Path(gamefile).parents[4])
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        self.env = textworld.start(str(gamefile), textworld.EnvInfos(won=True),
                                   wrappers=[AlfredDemangler])
        self.env.seed(seed)

    def reset(self):
        state = self.env.reset()
        return str(state.feedback), bool(state.won)

    def step(self, action):
        state, score, done = self.env.step(action)
        return str(state.feedback), float(score), bool(done), bool(state.won)

    def close(self):
        self.env.close()


def normalize_action(text):
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("```")]
    if not lines:
        return "__empty_action__"
    return re.sub(r"^(?:action\s*:\s*|>\s*)", "", lines[0], flags=re.I).strip()


class EpisodeRunner:
    def __init__(self, config, project, run_dir, client, env_factory=AlfworldEnvironment):
        self.config, self.project, self.directory, self.client = config, project, Path(run_dir), client
        self.env_factory = env_factory
        self.base_prompt = (project / "prompts/actor.txt").read_text()

    def messages(self, memory, initial, steps):
        system = self.base_prompt.replace("{memory}", memory.render())
        # Each call is stateless. Always preserve the full memory and initial task observation.
        retained = []
        limit = self.config["episode"]["history_tokens"]
        for step in reversed(steps):
            text = f"Action: {step['action']}\nObservation: {step['observation']}"
            if retained and self.client.tokens("\n".join([text, *retained])) > limit:
                break
            retained.insert(0, text)
        while True:
            omitted = len(steps) - len(retained)
            body = "Initial observation and task:\n" + initial
            if omitted:
                body += f"\n[{omitted} earlier steps omitted from the actor's context window; full trace is logged.]"
            body += "\nRecent interaction:\n" + ("\n".join(retained) or "(none)") + "\nNext command:"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": body}]
            c = self.config["model"]
            if self.client.chat_tokens(messages) + c["actor_output_tokens"] <= c["actor_context"]:
                return messages
            if not retained:
                raise ContextBudgetError("Full experience memory plus initial task cannot fit actor context")
            retained.pop(0)

    def run(self, task, memory, actor_seed, episode_id, stage):
        path = self.directory / "episodes" / (episode_id + ".json")
        progress_path = path.with_suffix(".progress.json")
        initial_hash = memory.fingerprint
        signature = digest({"task": task, "memory": memory.to_dict(), "seed": actor_seed,
                            "episode_config": self.config["episode"], "stage": stage})
        if path.exists():
            saved = read_json(path)
            if saved["input_hash"] != signature:
                raise InfrastructureError("Episode cache does not match current inputs: " + episode_id)
            return saved
        previous = read_json(progress_path) if progress_path.exists() else None
        if previous and previous["input_hash"] != signature:
            raise InfrastructureError("Partial episode does not match current inputs")
        game = (self.project / self.config["data_root"] / task["game_path"]).resolve()
        if hashlib.sha256(game.read_bytes()).hexdigest() != task["game_sha256"]:
            raise InfrastructureError("ALFWorld task file changed: " + task["id"])
        env_seed = seed_for(self.config["seed"], "environment", task["id"])
        try:
            env = self.env_factory(game, env_seed)
        except Exception as exc:
            raise InfrastructureError("Cannot create ALFWorld environment: " + str(exc)) from exc
        event(self.directory, "episode_start", episode_id=episode_id, stage=stage,
              task=task["id"], memory_hash=initial_hash, actor_seed=actor_seed, env_seed=env_seed)
        print(f"  {episode_id}: start (memory v{memory.version})", flush=True)
        try:
            initial, won = env.reset()
            if previous and initial != previous["initial_observation"]:
                raise InfrastructureError("Reset state differs from interrupted episode")
            trace = {"id": episode_id, "stage": stage, "task": task,
                     "input_hash": signature, "memory_hash": initial_hash,
                     "actor_seed": actor_seed, "environment_seed": env_seed,
                     "initial_observation": initial, "steps": [], "success": won,
                     "completed": False, "termination": None}
            for index in range(self.config["episode"]["max_steps"]):
                if won:
                    break
                result = self.client.call(self.messages(memory, initial, trace["steps"]),
                                          seed_for(actor_seed, "action", index),
                                          episode_id + f"/step-{index + 1}", actor=True)
                action = normalize_action(result["text"])
                try:
                    observation, score, done, won = env.step(action)
                except Exception as exc:
                    raise InfrastructureError("ALFWorld engine failed during step: " + str(exc)) from exc
                step = {"step": index + 1, "action": action, "raw_model_output": result["text"],
                        "model_call": result["call_hash"], "observation": observation,
                        "score": score, "done": done, "won": won}
                if previous and index < len(previous["steps"]):
                    old = previous["steps"][index]
                    if any(step[key] != old[key] for key in ("action", "observation", "won", "score", "done")):
                        raise InfrastructureError("Environment replay differs from partial episode")
                trace["steps"].append(step)
                trace["success"] = won
                write_json(progress_path, trace)
                event(self.directory, "step", episode_id=episode_id, step=index + 1,
                      action=action, won=won)
                if (index + 1) % 5 == 0:
                    print(f"    {episode_id}: step {index + 1}/{self.config['episode']['max_steps']}", flush=True)
                if done:
                    trace["termination"] = "won" if won else "environment_done"
                    break
            trace["termination"] = trace["termination"] or ("won" if won else "step_limit")
            trace["completed"] = True
            if memory.fingerprint != initial_hash:
                raise InfrastructureError("Memory changed during an episode")
            write_json(path, trace)
            progress_path.unlink(missing_ok=True)
            event(self.directory, "episode_end", episode_id=episode_id, success=won,
                  steps=len(trace["steps"]), termination=trace["termination"])
            print(f"  {episode_id}: {'success' if won else 'failure'} ({len(trace['steps'])} steps)", flush=True)
            return trace
        finally:
            env.close()
