"""On-call incidents. The agent may only read evidence through tools."""

from __future__ import annotations

import json
from pathlib import Path


CANDIDATES = [
    "payment_pool",
    "bad_deploy",
    "cache_stampede",
    "disk_full",
    "network_blip",
]

TOOLS = ("query_logs", "query_metrics", "query_changes")


def load_tasks(path: Path | None = None) -> list[dict]:
    file = path or Path(__file__).resolve().parents[2] / "data" / "incidents.json"
    return json.loads(file.read_text(encoding="utf-8"))


def observe(task: dict, tool: str, service: str) -> str | None:
    if tool not in TOOLS:
        return None
    if service not in task["services"]:
        return None
    return task["evidence"].get(f"{tool}:{service}", "无异常记录")


def useful_calls(task: dict) -> set[tuple[str, str]]:
    return {tuple(pair) for pair in task["useful_tools"]}
