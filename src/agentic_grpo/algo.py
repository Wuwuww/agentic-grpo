"""GRPO pieces aligned with verl's outcome advantage and clipped actor loss.

The formulas follow DeepSeek GRPO and verl's `compute_grpo_outcome_advantage` /
`compute_policy_loss` / Schulman low-variance KL. This file does not import verl.
Log-probabilities here are sequence scores from a small prompt policy, not token
log-probs of a GPU actor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def group_advantages(rewards: list[float], *, norm_by_std: bool = True, epsilon: float = 1e-6) -> list[float]:
    """Outcome advantage inside one prompt group.

    A group of size 1 uses mean 0 and std 1, matching verl. Larger groups subtract
    the group mean and, by default, divide by the unbiased standard deviation.
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [rewards[0] / (1.0 + epsilon)]
    mean = sum(rewards) / len(rewards)
    var = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
    std = math.sqrt(var)
    scale = (std + epsilon) if norm_by_std else 1.0
    return [(reward - mean) / scale for reward in rewards]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PolicyLoss:
    loss: float
    clipfrac: float
    approx_kl: float
    grad_coef: list[float]


def clipped_policy_loss(
    logprobs: list[float],
    old_logprobs: list[float],
    advantages: list[float],
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    clip_ratio_c: float = 3.0,
) -> PolicyLoss:
    """Sequence-level PPO clip. Gradient coefficient is d(loss)/d(logprob)."""
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be > 1")
    if not (len(logprobs) == len(old_logprobs) == len(advantages)):
        raise ValueError("logprobs, old_logprobs, and advantages must align")
    if not logprobs:
        return PolicyLoss(0.0, 0.0, 0.0, [])

    losses: list[float] = []
    coefs: list[float] = []
    clipped = 0
    kl_sum = 0.0
    for logprob, old, advantage in zip(logprobs, old_logprobs, advantages):
        delta = _clip(logprob - old, -20.0, 20.0)
        ratio = math.exp(delta)
        kl_sum += -delta
        unclipped = -advantage * ratio
        clipped_ratio = _clip(ratio, 1.0 - clip_low, 1.0 + clip_high)
        clipped_loss = -advantage * clipped_ratio
        # max(-ratio * A, -clip(ratio) * A), same selection as verl.
        if unclipped >= clipped_loss:
            chosen = unclipped
            coef = -advantage * ratio
        else:
            chosen = clipped_loss
            coef = 0.0
            clipped += 1
        if advantage < 0:
            dual = -advantage * clip_ratio_c
            if dual < chosen:
                chosen = dual
                coef = 0.0
                clipped += 1
        losses.append(chosen)
        coefs.append(coef)
    count = float(len(losses))
    return PolicyLoss(
        loss=sum(losses) / count,
        clipfrac=clipped / count,
        approx_kl=kl_sum / count,
        grad_coef=coefs,
    )


def low_var_kl(logprob: float, ref_logprob: float) -> tuple[float, float]:
    """Schulman k3. Returns (kl, d(kl)/d(logprob))."""
    kl_delta = _clip(ref_logprob - logprob, -20.0, 20.0)
    ratio = math.exp(kl_delta)
    value = _clip(ratio - kl_delta - 1.0, -10.0, 10.0)
    # d(kl_delta)/d(logprob) = -1, d(ratio - kl_delta - 1)/d(kl_delta) = ratio - 1
    grad = -(ratio - 1.0)
    return value, grad
