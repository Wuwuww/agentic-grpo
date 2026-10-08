import math
from pathlib import Path

from agentic_grpo.search.corpus import Retriever, load_passages, load_questions
from agentic_grpo.search.em import exact_match, normalize_answer
from agentic_grpo.search.policy import SearchPolicy, action_mass
from agentic_grpo.search.rollout import ScriptedSearch, Span, loss_mask, rollout
from agentic_grpo.search.trainer import fit, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_exact_match_normalizes_case_and_articles():
    assert normalize_answer("The Beijing.") == "beijing"
    assert exact_match(" Beijing ", ["beijing"]) == 1.0
    assert exact_match("network_blip", ["payment_pool"]) == 0.0
    assert exact_match(None, ["payment_pool"]) == 0.0


def test_retriever_ranks_the_runbook_that_contains_the_answer():
    passages = load_passages()
    retriever = Retriever(passages)
    questions = load_questions()
    for task in questions:
        if "network_blip" in task["question"]:
            continue
        hits = retriever.search(task["question"], topk=3)
        assert hits
        blob = " ".join(item["text"] for item in hits)
        assert task["golden_answers"][0] in blob
        assert hits[0]["text"].find(task["golden_answers"][0]) >= 0


def test_information_tokens_are_masked_out_of_the_loss():
    spans = [
        Span("<think>查一下。</think><search>checkout 5xx</search>", "llm"),
        Span("<information> alpha beta gamma delta </information>", "retrieved"),
        Span("<answer> payment_pool </answer>", "llm"),
    ]
    mask = loss_mask(spans)
    assert 0 in mask
    assert mask[0] == 1
    assert mask[-1] == 1
    assert mask[mask.index(0) :].count(0) >= 4


def test_retrieved_length_does_not_change_the_masked_search_gradient():
    short = {"search_calls": 1.0, "info_tokens": 3.0}
    long = {"search_calls": 1.0, "info_tokens": 400.0}
    state = SearchPolicy(search_rate=0.4)
    assert state.logprob(short, mask_retrieved=True) == state.logprob(long, mask_retrieved=True)
    assert action_mass(long, mask_retrieved=False) > action_mass(short, mask_retrieved=False)


class _SearchThenAnswer:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def complete(self, messages, *, sample_index: int = 0) -> str:
        self.calls += 1
        if self.calls == 1:
            return "<think>题面不够。</think>\n<search>checkout 5xx</search>"
        return f"<think>手册写了。</think>\n<answer> {self.answer} </answer>"


def test_rollout_reward_is_exact_match_and_inserts_information():
    task = load_questions()[0]
    retriever = Retriever(load_passages())
    good = rollout(task, SearchPolicy(), _SearchThenAnswer("payment_pool"), retriever, 0)
    bad = rollout(task, SearchPolicy(), _SearchThenAnswer("network_blip"), retriever, 0)
    assert good.reward == 1.0
    assert bad.reward == 0.0
    assert any(span.source == "retrieved" for span in good.spans)
    assert "<information>" in "".join(span.text for span in good.spans if span.source == "retrieved")
    assert 0 in loss_mask(good.spans)


def test_scripted_search_r1_raises_exact_match_without_weight_updates(tmp_path):
    config = load_config(ROOT / "configs" / "search_r1_grpo.json")
    questions = load_questions()
    history = fit(
        config,
        ScriptedSearch(questions, group_n=config["actor_rollout_ref"]["rollout"]["n_agent"]),
        questions,
        Retriever(load_passages()),
        tmp_path,
    )
    assert history[0]["llm_weight_update"] == "skipped"
    assert history[0]["reward_type"] == "exact_match"
    assert history[0]["state_masking"] is True
    assert history[-1]["em_rate"] > history[0]["em_rate"]
    assert history[-1]["search_rate"] > config["policy"]["search_rate"]
    assert history[-1]["em_rate"] == 1.0
    assert math.isclose(history[-1]["search_call_mean"], 0.8, abs_tol=1e-6)
