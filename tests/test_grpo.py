import json
import math
from pathlib import Path

from agentic_grpo.algo import clipped_policy_loss, group_advantages, low_var_kl
from agentic_grpo.chat import APIChat, ScriptedChat
from agentic_grpo.reward import score_trajectory
from agentic_grpo.tasks import load_tasks, observe
from agentic_grpo.trainer import fit, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_group_advantage_matches_verl_singleton_and_std():
    assert group_advantages([2.0])[0] == 2.0 / (1.0 + 1e-6)
    advantages = group_advantages([1.0, 0.0, 0.0, 0.0])
    assert advantages[0] > 0
    assert all(item < 0 for item in advantages[1:])
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-6)
    assert math.isclose(advantages[0], 1.5, abs_tol=1e-4)


def test_low_var_kl_is_zero_at_equality():
    value, grad = low_var_kl(0.0, 0.0)
    assert value == 0.0
    assert grad == 0.0


def test_clip_blocks_gradient_when_ratio_is_too_large():
    loss = clipped_policy_loss([math.log(2.0)], [0.0], [1.0], clip_low=0.2, clip_high=0.2)
    assert math.isclose(loss.loss, -1.2, abs_tol=1e-6)
    assert loss.grad_coef == [0.0]
    assert loss.clipfrac == 1.0


def test_reward_pays_for_evidence_and_root_cause():
    task = load_tasks()[0]
    assert observe(task, "query_logs", "payment") == "database connection pool exhausted"
    good = score_trajectory(task, [("query_logs", "payment")], "payment_pool", 0)
    guess = score_trajectory(task, [], "network_blip", 0)
    assert good > 1.0
    assert guess < 0


def test_scripted_training_raises_reward_without_touching_llm_weights(tmp_path):
    config = load_config(ROOT / "configs" / "oncall_grpo.json")
    tasks = load_tasks()
    history = fit(config, ScriptedChat(tasks), tasks, tmp_path)
    assert history[0]["llm_weight_update"] == "skipped"
    assert history[-1]["reward_mean"] > history[0]["reward_mean"]
    assert history[-1]["investigate"] > config["policy"]["investigate"]
    saved = json.loads((tmp_path / "policy_params.json").read_text(encoding="utf-8"))
    assert saved["step"] >= 1


def test_api_payload_includes_policy_param(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "Final: disk_full"}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    chat = APIChat("https://example.invalid/v1", "test-key", "fake-model")
    text = chat.complete(
        [
            {"role": "system", "content": "policy_param investigate=0.500000"},
            {"role": "user", "content": "告警: monitoring 写入失败"},
        ],
        sample_index=1,
    )
    assert text == "Final: disk_full"
    assert "investigate=0.500000" in captured["body"]["messages"][0]["content"]
    assert captured["body"]["temperature"] == 0.75
