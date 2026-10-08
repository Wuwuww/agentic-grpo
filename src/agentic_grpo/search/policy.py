"""Prompt-side search_rate. Retrieved tokens are excluded when state masking is on."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SearchPolicy:
    search_rate: float = 0.05
    step: int = 0

    def logprob(self, features: dict[str, float], *, mask_retrieved: bool) -> float:
        return self.search_rate * action_mass(features, mask_retrieved=mask_retrieved)

    def feature_grad(self, features: dict[str, float], *, mask_retrieved: bool) -> dict[str, float]:
        return {"search_rate": action_mass(features, mask_retrieved=mask_retrieved)}

    def add_grad(self, grad: dict[str, float], lr: float) -> None:
        updated = self.search_rate - lr * grad["search_rate"]
        self.search_rate = min(3.0, max(0.0, updated))
        self.step += 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def action_mass(features: dict[str, float], *, mask_retrieved: bool) -> float:
    """LLM search actions, plus retrieved length only when masking is off.

    Search-R1 sets I(y_t)=0 on tokens inside <information>. Counting those
    tokens as policy actions makes the gradient track passage length.
    """
    mass = float(features.get("search_calls", 0.0))
    if not mask_retrieved:
        mass += float(features.get("info_tokens", 0.0))
    return mass


def render_system(state: SearchPolicy) -> str:
    return (
        "模型权重不会被更新。下面的数是 GRPO 写进提示里的策略参数。\n"
        f"policy_param search_rate={state.search_rate:.6f}\n"
        "search_rate 越高，题面里没有答案时越应该发出 <search>，而不是直接猜。\n"
        "<information> 里的文字是检索引擎写回的环境文本。"
    )
