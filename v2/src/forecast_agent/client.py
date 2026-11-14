import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from tokenizers import Tokenizer

from .io import digest, dumps, event, read_json, write_json


class Ollama:
    def __init__(self, config, project, directory):
        self.config = config["model"]
        self.directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(project / config["tokenizer"]))
        self.http = build_opener(ProxyHandler({}))
        self.embedding_path = self.directory / "embedding_cache.json"
        self.embedding_cache = read_json(self.embedding_path) if self.embedding_path.exists() else {}

    def request(self, endpoint, payload=None):
        request = Request(self.config["url"].rstrip("/") + "/api/" + endpoint,
                          data=dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        with self.http.open(request, timeout=self.config["timeout"]) as response:
            return json.load(response)

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def message_tokens(self, messages):
        text = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
                       for m in messages)
        return self.tokens(text + "<|im_start|>assistant\n") + 64

    def chat(self, role, messages, schema, seed, purpose):
        settings = self.config["roles"][role]
        size = self.message_tokens(messages)
        if size + settings["output"] > settings["context"]:
            raise ValueError(f"{purpose}: input {size} + output {settings['output']} exceeds "
                             f"context {settings['context']}; experience has not been truncated")
        request = {
            "model": self.config["name"], "messages": messages, "format": schema,
            "stream": False, "think": False, "keep_alive": self.config["keep_alive"],
            "options": {"num_ctx": settings["context"], "num_predict": settings["output"],
                        "temperature": settings["temperature"], "seed": seed,
                        "num_batch": self.config["batch_size"]},
        }
        path = self.directory / "model_calls" / (digest([purpose, request]) + ".json")
        if path.exists():
            saved = read_json(path)
            response = saved["response"]
        else:
            response = self.request("chat", request)
            write_json(path, {"role": role, "purpose": purpose, "request": request, "response": response})
        if response["done_reason"] == "length":
            raise ValueError(f"{purpose}: model output reached the configured output limit")
        event(self.directory, "model_call", role=role, purpose=purpose,
              prompt_tokens=response.get("prompt_eval_count", 0), output_tokens=response.get("eval_count", 0))
        return json.loads(response["message"]["content"])

    def embed(self, texts):
        missing = list(dict.fromkeys(t for t in texts if digest(t) not in self.embedding_cache))
        if missing:
            response = self.request("embed", {"model": self.config["embedding"], "input": missing,
                                               "truncate": False, "keep_alive": 0})
            self.embedding_cache.update({digest(t): v for t, v in zip(missing, response["embeddings"])})
            write_json(self.embedding_path, self.embedding_cache)
        return np.asarray([self.embedding_cache[digest(t)] for t in texts])

    def metadata(self):
        return {"server": self.request("version"), "models": self.request("tags")["models"]}
