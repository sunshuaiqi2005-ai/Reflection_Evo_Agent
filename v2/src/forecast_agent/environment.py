import os
from pathlib import Path


def task_goal(initial_observation):
    return initial_observation.split("Your task is to: ", 1)[1]


class Alfworld:
    def __init__(self, gamefile, seed):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        os.environ["ALFWORLD_DATA"] = str(Path(gamefile).parents[4])
        self.env = textworld.start(str(gamefile), textworld.EnvInfos(won=True), wrappers=[AlfredDemangler])
        self.env.seed(seed)

    def reset(self):
        state = self.env.reset()
        return str(state.feedback), bool(state.won)

    def step(self, action):
        state, score, done = self.env.step(action)
        return str(state.feedback), float(score), bool(done), bool(state.won)

    def close(self):
        self.env.close()
