"""verl-shaped fit loop: group rollout, GRPO advantage, clipped update on prompt params."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from agentic_grpo.algo import clipped_policy_loss, group_advantages, low_var_kl
from agentic_grpo.chat import APIChat, ScriptedChat
from agentic_grpo.policy import PolicyState
from agentic_grpo.rollout import Trajectory, rollout
from agentic_grpo.tasks import load_tasks


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fit(config: dict, chat, tasks: list[dict], output_dir: Path) -> list[dict]:
    state = PolicyState(
        investigate=float(config["policy"]["investigate"]),
        commit=float(config["policy"]["commit"]),
    )
    reference = PolicyState(investigate=state.investigate, commit=state.commit)
    group_n = int(config["actor_rollout_ref"]["rollout"]["n"])
    if group_n < 2:
        raise ValueError("GRPO 需要 actor_rollout_ref.rollout.n >= 2")
    epochs = int(config["trainer"]["total_epochs"])
    ppo_epochs = int(config["actor_rollout_ref"]["actor"]["ppo_epochs"])
    lr = float(config["actor_rollout_ref"]["actor"]["optim"]["lr"])
    clip_low = float(config["actor_rollout_ref"]["actor"]["clip_ratio_low"])
    clip_high = float(config["actor_rollout_ref"]["actor"]["clip_ratio_high"])
    kl_coef = float(config["actor_rollout_ref"]["actor"]["kl_loss_coef"])
    use_kl = bool(config["actor_rollout_ref"]["actor"]["use_kl_loss"])
    norm_std = bool(config["algorithm"]["norm_adv_by_std_in_grpo"])
    max_steps = int(config["rollout"]["max_steps"])
    history: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        grouped: list[list[Trajectory]] = []
        for task in tasks:
            group = [
                rollout(task, state, chat, sample_index=index, max_steps=max_steps)
                for index in range(group_n)
            ]
            grouped.append(group)
        advantages: list[float] = []
        flat: list[Trajectory] = []
        for group in grouped:
            scores = group_advantages([item.reward for item in group], norm_by_std=norm_std)
            advantages.extend(scores)
            flat.extend(group)
        old_logps = [state.logprob(item.features) for item in flat]
        ref_logps = [reference.logprob(item.features) for item in flat]
        metrics = {
            "epoch": epoch,
            "reward_mean": _mean([item.reward for item in flat]),
            "correct_rate": _mean([item.features["correct"] for item in flat]),
            "llm_weight_update": "skipped",
        }
        for _ in range(ppo_epochs):
            logps = [state.logprob(item.features) for item in flat]
            policy = clipped_policy_loss(
                logps,
                old_logps,
                advantages,
                clip_low=clip_low,
                clip_high=clip_high,
            )
            kl_values = []
            coefs = list(policy.grad_coef)
            if use_kl:
                for index, logp in enumerate(logps):
                    value, grad = low_var_kl(logp, ref_logps[index])
                    kl_values.append(value)
                    coefs[index] += kl_coef * grad
            else:
                kl_values = [0.0 for _ in logps]
            grad = {"investigate": 0.0, "commit": 0.0}
            for traj, coef in zip(flat, coefs):
                local = state.feature_grad(traj.features)
                grad["investigate"] += coef * local["investigate"]
                grad["commit"] += coef * local["commit"]
            count = float(len(flat))
            grad = {key: value / count for key, value in grad.items()}
            state.add_grad(grad, lr)
            metrics.update(
                {
                    "pg_loss": policy.loss + (kl_coef * _mean(kl_values) if use_kl else 0.0),
                    "pg_clipfrac": policy.clipfrac,
                    "ppo_kl": policy.approx_kl,
                    "kl_loss": _mean(kl_values),
                    "grad_norm": math.sqrt(sum(value * value for value in grad.values())),
                    "investigate": state.investigate,
                    "commit": state.commit,
                    "prompt_policy_update": "applied",
                }
            )
        history.append(metrics)
        print(
            f"epoch={epoch} reward_mean={metrics['reward_mean']:.3f} "
            f"correct_rate={metrics['correct_rate']:.3f} pg_loss={metrics['pg_loss']:.3f} "
            f"investigate={state.investigate:.3f} llm_weights=skipped"
        )
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history),
        encoding="utf-8",
    )
    state.save(output_dir / "policy_params.json")
    return history


def build_chat(config: dict, tasks: list[dict], backend: str):
    if backend == "scripted":
        return ScriptedChat(tasks)
    if backend == "api":
        actor = config["actor_rollout_ref"]
        return APIChat.from_env(actor["model"]["name"], float(actor["rollout"]["temperature"]))
    raise ValueError(f"未知后端: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="On-call GRPO loop with a frozen API policy")
    parser.add_argument("--config", type=Path, default=Path("configs/oncall_grpo.json"))
    parser.add_argument("--backend", choices=("scripted", "api"), default="scripted")
    parser.add_argument("--output", type=Path, default=Path("runs/oncall"))
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = load_tasks()
    fit(config, build_chat(config, tasks, args.backend), tasks, args.output)


if __name__ == "__main__":
    main()
