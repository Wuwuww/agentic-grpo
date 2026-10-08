"""Search-R1 shaped loop: GRPO on outcome EM, with retrieved-token masking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_grpo.algo import clipped_policy_loss, group_advantages, low_var_kl
from agentic_grpo.chat import APIChat
from agentic_grpo.search.corpus import Retriever, load_passages, load_questions
from agentic_grpo.search.policy import SearchPolicy
from agentic_grpo.search.rollout import ScriptedSearch, loss_mask, rollout


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fit(config: dict, chat, questions: list[dict], retriever: Retriever, output_dir: Path) -> list[dict]:
    state = SearchPolicy(search_rate=float(config["policy"]["search_rate"]))
    reference = SearchPolicy(search_rate=state.search_rate)
    group_n = int(config["actor_rollout_ref"]["rollout"]["n_agent"])
    if group_n < 2:
        raise ValueError("GRPO 需要 actor_rollout_ref.rollout.n_agent >= 2")
    epochs = int(config["trainer"]["total_epochs"])
    ppo_epochs = int(config["actor_rollout_ref"]["actor"]["ppo_epochs"])
    lr = float(config["actor_rollout_ref"]["actor"]["optim"]["lr"])
    clip_low = float(config["actor_rollout_ref"]["actor"]["clip_ratio_low"])
    clip_high = float(config["actor_rollout_ref"]["actor"]["clip_ratio_high"])
    kl_coef = float(config["actor_rollout_ref"]["actor"]["kl_loss_coef"])
    use_kl = bool(config["actor_rollout_ref"]["actor"]["use_kl_loss"])
    mask_retrieved = bool(config["actor_rollout_ref"]["actor"]["state_masking"])
    norm_std = bool(config["algorithm"]["norm_adv_by_std_in_grpo"])
    max_turns = int(config["rollout"]["max_turns"])
    topk = int(config["rollout"]["topk"])
    history: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        grouped = []
        for task in questions:
            group = [
                rollout(
                    task,
                    state,
                    chat,
                    retriever,
                    sample_index=index,
                    max_turns=max_turns,
                    topk=topk,
                )
                for index in range(group_n)
            ]
            grouped.append(group)
        advantages: list[float] = []
        flat = []
        for group in grouped:
            advantages.extend(group_advantages([item.reward for item in group], norm_by_std=norm_std))
            flat.extend(group)
        old_logps = [state.logprob(item.features, mask_retrieved=mask_retrieved) for item in flat]
        ref_logps = [reference.logprob(item.features, mask_retrieved=mask_retrieved) for item in flat]
        masks = [loss_mask(item.spans) for item in flat]
        mask_bits = [bit for mask in masks for bit in mask]
        metrics = {
            "epoch": epoch,
            "em_rate": _mean([item.reward for item in flat]),
            "search_call_mean": _mean([item.features["search_calls"] for item in flat]),
            "masked_token_frac": 1.0 - (_mean([float(bit) for bit in mask_bits]) if mask_bits else 0.0),
            "state_masking": mask_retrieved,
            "reward_type": "exact_match",
            "llm_weight_update": "skipped",
        }
        for _ in range(ppo_epochs):
            logps = [state.logprob(item.features, mask_retrieved=mask_retrieved) for item in flat]
            policy = clipped_policy_loss(logps, old_logps, advantages, clip_low=clip_low, clip_high=clip_high)
            coefs = list(policy.grad_coef)
            kl_values = []
            if use_kl:
                for index, logp in enumerate(logps):
                    value, grad = low_var_kl(logp, ref_logps[index])
                    kl_values.append(value)
                    coefs[index] += kl_coef * grad
            else:
                kl_values = [0.0 for _ in logps]
            grad = {"search_rate": 0.0}
            for traj, coef in zip(flat, coefs):
                local = state.feature_grad(traj.features, mask_retrieved=mask_retrieved)
                grad["search_rate"] += coef * local["search_rate"]
            count = float(len(flat))
            grad["search_rate"] /= count
            state.add_grad(grad, lr)
            metrics.update(
                {
                    "pg_loss": policy.loss + (kl_coef * _mean(kl_values) if use_kl else 0.0),
                    "pg_clipfrac": policy.clipfrac,
                    "kl_loss": _mean(kl_values),
                    "grad_norm": abs(grad["search_rate"]),
                    "search_rate": state.search_rate,
                    "prompt_policy_update": "applied",
                }
            )
        history.append(metrics)
        print(
            f"epoch={epoch} em_rate={metrics['em_rate']:.3f} "
            f"search_calls={metrics['search_call_mean']:.3f} "
            f"search_rate={state.search_rate:.3f} "
            f"masked_token_frac={metrics['masked_token_frac']:.3f} llm_weights=skipped"
        )
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history),
        encoding="utf-8",
    )
    state.save(output_dir / "policy_params.json")
    return history


def build_chat(config: dict, questions: list[dict], backend: str):
    group_n = int(config["actor_rollout_ref"]["rollout"]["n_agent"])
    if backend == "scripted":
        return ScriptedSearch(questions, group_n=group_n)
    if backend == "api":
        actor = config["actor_rollout_ref"]
        return APIChat.from_env(actor["model"]["name"], float(actor["rollout"]["temperature"]))
    raise ValueError(f"未知后端: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-R1 loop over runbook QA, without LLM weight updates")
    parser.add_argument("--config", type=Path, default=Path("configs/search_r1_grpo.json"))
    parser.add_argument("--backend", choices=("scripted", "api"), default="scripted")
    parser.add_argument("--output", type=Path, default=Path("runs/search-r1"))
    args = parser.parse_args()
    config = load_config(args.config)
    questions = load_questions()
    retriever = Retriever(load_passages())
    fit(config, build_chat(config, questions, args.backend), questions, retriever, args.output)


if __name__ == "__main__":
    main()
