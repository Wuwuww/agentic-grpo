"""Chat backends. ScriptedChat explores; APIChat calls an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from urllib.error import URLError

from agentic_grpo.tasks import useful_calls

INVESTIGATE_RE = re.compile(r"investigate=([+-]?\d+(?:\.\d+)?)")


class ScriptedChat:
    """Deterministic explorer. One sample in each group investigates even at low θ."""

    def __init__(self, tasks: list[dict]):
        self.by_alert = {task["alert"]: task for task in tasks}

    def complete(self, messages: list[dict], *, sample_index: int = 0) -> str:
        system = messages[0]["content"]
        user = next(message["content"] for message in messages if message["role"] == "user")
        match = re.search(r"告警:\s*(.+)", user)
        alert = match.group(1).strip() if match else ""
        task = self.by_alert[alert]
        investigate = float(INVESTIGATE_RE.search(system).group(1))
        called = _called_tools(messages)
        useful = list(useful_calls(task))
        group_n = 4
        draw = (sample_index + 1) / (group_n + 1)
        explore = 0.5 if sample_index % group_n == 0 else 0.0
        careful = investigate + explore > draw
        remaining = [pair for pair in useful if pair not in called]
        if careful and remaining:
            tool, service = remaining[0]
            return f"Action: {tool} {service}"
        if careful:
            return f"Final: {task['root_cause']}"
        return "Final: network_blip"


class APIChat:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_env(cls, model: str, temperature: float) -> "APIChat":
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        base_url = os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if not api_key:
            raise RuntimeError("API 后端需要环境变量 LLM_API_KEY 或 OPENAI_API_KEY")
        return cls(base_url, api_key, model, temperature)

    def complete(self, messages: list[dict], *, sample_index: int = 0) -> str:
        payload = {
            "model": self.model,
            "temperature": min(1.2, self.temperature + 0.05 * sample_index),
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"LLM 请求失败: {exc}") from exc
        return body["choices"][0]["message"]["content"].strip()


def _called_tools(messages: list[dict]) -> set[tuple[str, str]]:
    called: set[tuple[str, str]] = set()
    for message in messages:
        if message["role"] != "assistant":
            continue
        match = re.search(r"Action:\s*(query_logs|query_metrics|query_changes)\s+([A-Za-z0-9_-]+)", message["content"])
        if match:
            called.add((match.group(1).lower(), match.group(2)))
    return called
