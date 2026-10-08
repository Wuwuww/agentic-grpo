"""Verifiable reward for one on-call trajectory. No learned reward model."""

from __future__ import annotations

from agentic_grpo.tasks import useful_calls


def score_trajectory(task: dict, tool_calls: list[tuple[str, str]], final: str | None, invalid: int) -> float:
    reward = 0.0
    if final is None:
        reward -= 0.5
    elif final == task["root_cause"]:
        reward += 1.0
    else:
        reward -= 0.2
    useful = useful_calls(task)
    called = set(tool_calls)
    reward += 0.2 * len(useful & called)
    reward -= 0.1 * len(called - useful)
    reward -= 0.3 * invalid
    return reward


def trajectory_features(task: dict, tool_calls: list[tuple[str, str]], final: str | None, invalid: int) -> dict[str, float]:
    useful = useful_calls(task)
    called = set(tool_calls)
    return {
        "useful_hits": float(len(useful & called)),
        "correct": 1.0 if final == task["root_cause"] else 0.0,
        "useless_calls": float(len(called - useful)),
        "invalid": float(invalid),
    }
