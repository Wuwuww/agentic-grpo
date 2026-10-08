"""Prompt-side parameters. LLM weights stay frozen; these numbers are what GRPO steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


FEATURES = ("useful_hits", "correct", "useless_calls", "invalid")


@dataclass
class PolicyState:
    investigate: float = 0.05
    commit: float = 0.0
    step: int = 0

    def logprob(self, features: dict[str, float]) -> float:
        return (
            self.investigate * features.get("useful_hits", 0.0)
            + self.commit * features.get("correct", 0.0)
            - 0.25 * self.investigate * features.get("useless_calls", 0.0)
            - self.commit * features.get("invalid", 0.0)
        )

    def feature_grad(self, features: dict[str, float]) -> dict[str, float]:
        """d(logprob)/d(parameter)."""
        return {
            "investigate": features.get("useful_hits", 0.0) - 0.25 * features.get("useless_calls", 0.0),
            "commit": features.get("correct", 0.0) - features.get("invalid", 0.0),
        }

    def add_grad(self, grad: dict[str, float], lr: float) -> None:
        # Loss gradient is passed in; the caller already negated the policy objective.
        self.investigate -= lr * grad["investigate"]
        self.commit -= lr * grad["commit"]
        self.step += 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def render_system(state: PolicyState) -> str:
    return (
        "你是值班排障代理。模型权重不会被更新；下面两个数是 GRPO 写进提示里的策略参数。\n"
        f"policy_param investigate={state.investigate:.6f}\n"
        f"policy_param commit={state.commit:.6f}\n"
        "investigate 越高，越要先调用工具拿到能区分根因的证据，再下结论。\n"
        "commit 越高，证据足够时就给出唯一根因 id，不要改口。\n"
        "每一轮只输出一行，格式只能是：\n"
        "Action: query_logs <service>\n"
        "Action: query_metrics <service>\n"
        "Action: query_changes <service>\n"
        "Final: <root_cause_id>\n"
        "不要输出其它文字。"
    )
