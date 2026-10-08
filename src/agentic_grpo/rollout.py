"""Multi-step tool rollout. The chat backend is either the API or a scripted stand-in."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_grpo.policy import PolicyState, render_system
from agentic_grpo.reward import score_trajectory, trajectory_features
from agentic_grpo.tasks import CANDIDATES, observe

ACTION_RE = re.compile(r"Action:\s*(query_logs|query_metrics|query_changes)\s+([A-Za-z0-9_-]+)", re.I)
FINAL_RE = re.compile(r"Final:\s*([A-Za-z0-9_-]+)", re.I)


@dataclass
class Trajectory:
    task_id: str
    sample_index: int
    tool_calls: list[tuple[str, str]]
    final: str | None
    invalid: int
    reward: float
    features: dict[str, float]
    transcript: list[str] = field(default_factory=list)


def user_prompt(task: dict) -> str:
    services = ", ".join(task["services"])
    candidates = ", ".join(CANDIDATES)
    return (
        f"告警: {task['alert']}\n"
        f"可查服务: {services}\n"
        f"根因候选: {candidates}\n"
        "先取证，再给出一个根因 id。"
    )


def parse_decision(text: str) -> tuple[str, str] | tuple[None, str] | None:
    final = FINAL_RE.search(text or "")
    action = ACTION_RE.search(text or "")
    if final and (action is None or final.start() <= action.start()):
        return None, final.group(1)
    if action:
        return action.group(1).lower(), action.group(2)
    return None


def rollout(task: dict, state: PolicyState, chat, sample_index: int, max_steps: int = 4) -> Trajectory:
    messages = [
        {"role": "system", "content": render_system(state)},
        {"role": "user", "content": user_prompt(task)},
    ]
    tool_calls: list[tuple[str, str]] = []
    invalid = 0
    final: str | None = None
    transcript: list[str] = []
    for _ in range(max_steps):
        text = chat.complete(messages, sample_index=sample_index)
        transcript.append(text)
        messages.append({"role": "assistant", "content": text})
        decision = parse_decision(text)
        if decision is None:
            invalid += 1
            messages.append({"role": "user", "content": "Observation: 无法解析。请只输出 Action 或 Final 一行。"})
            continue
        tool, arg = decision
        if tool is None:
            final = arg
            break
        evidence = observe(task, tool, arg)
        if evidence is None:
            invalid += 1
            messages.append({"role": "user", "content": "Observation: 非法工具或服务。"})
            continue
        tool_calls.append((tool, arg))
        messages.append({"role": "user", "content": f"Observation: {evidence}"})
    features = trajectory_features(task, tool_calls, final, invalid)
    return Trajectory(
        task_id=task["id"],
        sample_index=sample_index,
        tool_calls=tool_calls,
        final=final,
        invalid=invalid,
        reward=score_trajectory(task, tool_calls, final, invalid),
        features=features,
        transcript=transcript,
    )
