from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler

import jsonschema
from tokenizers import Tokenizer

from .util import (ContextBudgetError, InfrastructureError, ProtocolError, digest,
                   event, read_json, write_json)


class OllamaClient:
    def __init__(self, config, project, run_dir):
        self.config = config["model"]
        self.run_dir = Path(run_dir)
        self.url = self.config["url"].rstrip("/")
        parsed = urlparse(self.url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("此实验只允许连接本机 Ollama。")
        if ":cloud" in self.config["name"]:
            raise ValueError("本地实验不能使用云模型。")
        self.http = build_opener(ProxyHandler({}))
        tokenizer_path = project / config["tokenizer"]
        if not tokenizer_path.is_file():
            raise InfrastructureError("缺少 Qwen 分词器，请先运行安装环境.command。")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.cache = self.run_dir / "model_calls"
        self.cache.mkdir(parents=True, exist_ok=True)

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def request(self, endpoint, payload=None, timeout=10):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + endpoint, data=body, headers={"Content-Type": "application/json"})
        with self.http.open(request, timeout=timeout) as response:
            return json.load(response)

    def chat_tokens(self, messages):
        chatml = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        # Reserve space for the exact Ollama template and the disabled-thinking prefix.
        return self.tokens(chatml + "<|im_start|>assistant\n") + 64

    def metadata(self):
        try:
            response = self.request("/api/tags")
            target = self.config["name"]
            if ":" not in target:
                target += ":latest"
            matches = [m for m in response["models"] if m["name"] == target]
            if not matches:
                raise InfrastructureError("找不到本地模型：" + target)
            return {"name": target, "digest": matches[0]["digest"],
                    "details": matches[0].get("details", {})}
        except (OSError, ValueError, KeyError) as exc:
            raise InfrastructureError("本地 Ollama 不可用，请先启动 Qwen 服务。") from exc

    def call(self, messages, seed, purpose, structured_schema=None, actor=False):
        context = self.config["actor_context" if actor else "reflection_context"]
        output = self.config["actor_output_tokens" if actor else "reflection_output_tokens"]
        estimated = self.chat_tokens(messages)
        if estimated + output > context:
            raise ContextBudgetError(f"{purpose}: complete prompt {estimated} + output {output} > {context}; no silent truncation")
        request = {"model": self.config["name"], "messages": messages, "stream": False,
                   "think": False, "keep_alive": "2m", "options": {
                       "num_ctx": context, "num_batch": 128, "num_predict": output,
                       "temperature": self.config["actor_temperature" if actor else "reflection_temperature"],
                       "top_p": 0.8, "top_k": 20, "seed": seed}}
        if structured_schema is not None:
            request["format"] = structured_schema
        key = digest(request)
        path = self.cache / (key + ".json")
        if path.exists():
            saved = read_json(path)
            if saved["request"] != request:
                raise InfrastructureError("Model-call cache hash mismatch")
            response = saved["response"]
        else:
            error = None
            for attempt in range(self.config["retries"] + 1):
                try:
                    response = self.request("/api/chat", request, self.config["timeout_seconds"])
                    if "error" in response:
                        raise InfrastructureError(response["error"])
                    break
                except (OSError, ValueError) as exc:
                    error = exc
                    if attempt < self.config["retries"]:
                        time.sleep(min(2 ** attempt, 4))
            else:
                raise InfrastructureError(f"Ollama request failed: {purpose}: {error}") from error
            write_json(path, {"purpose": purpose, "request": request, "response": response,
                              "input_token_bound": estimated})
        event(self.run_dir, "model_call", purpose=purpose, call_hash=key, seed=seed,
              prompt_tokens=response.get("prompt_eval_count"), output_tokens=response.get("eval_count"))
        if response.get("done") is not True:
            raise InfrastructureError(f"{purpose}: incomplete Ollama response")
        text = response.get("message", {}).get("content", "").strip()
        if not text or response.get("done_reason") == "length":
            raise ProtocolError(f"{purpose}: empty or truncated model output; see model_calls/{key}.json")
        if structured_schema is not None:
            try:
                parsed = json.loads(text)
                jsonschema.validate(parsed, structured_schema)
            except (ValueError, jsonschema.ValidationError) as exc:
                raise ProtocolError(f"{purpose}: invalid structured output") from exc
            return parsed
        return {"text": text, "call_hash": key}
