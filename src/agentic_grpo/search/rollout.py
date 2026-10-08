"""Interleaved <think> / <search> / <information> / <answer> rollout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_grpo.search.corpus import Retriever, format_passages
from agentic_grpo.search.em import contains_answer, exact_match
from agentic_grpo.search.policy import SearchPolicy, render_system

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
MASK_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)


@dataclass
class Span:
    text: str
    source: str  # llm | retrieved | env


@dataclass
class SearchTrajectory:
    task_id: str
    sample_index: int
    spans: list[Span]
    answer: str | None
    reward: float
    features: dict[str, float]
    transcript: list[str] = field(default_factory=list)


def user_prompt(question: str) -> str:
    return TEMPLATE.format(question=question)


def mask_tokens(text: str) -> list[str]:
    return MASK_TOKEN_RE.findall(text)


def loss_mask(spans: list[Span]) -> list[int]:
    """I(y_t)=1 for LLM tokens and 0 for retrieved or environment tokens."""
    mask: list[int] = []
    for span in spans:
        bit = 1 if span.source == "llm" else 0
        mask.extend([bit] * len(mask_tokens(span.text)))
    return mask


def parse_action(text: str) -> tuple[str, str] | None:
    answer = ANSWER_RE.search(text or "")
    search = SEARCH_RE.search(text or "")
    if answer and (search is None or answer.start() <= search.start()):
        return "answer", answer.group(1).strip()
    if search and search.group(1).strip():
        return "search", search.group(1).strip()
    return None


def trajectory_features(spans: list[Span], em: float) -> dict[str, float]:
    search_calls = 0
    info_tokens = 0
    for span in spans:
        if span.source == "llm":
            search_calls += len(SEARCH_RE.findall(span.text))
        elif span.source == "retrieved":
            info_tokens += len(mask_tokens(span.text))
    return {
        "search_calls": float(search_calls),
        "info_tokens": float(info_tokens),
        "em": em,
    }


def rollout(
    task: dict,
    state: SearchPolicy,
    chat,
    retriever: Retriever,
    sample_index: int,
    *,
    max_turns: int = 3,
    topk: int = 3,
) -> SearchTrajectory:
    messages = [
        {"role": "system", "content": render_system(state)},
        {"role": "user", "content": user_prompt(task["question"])},
    ]
    spans: list[Span] = []
    transcript: list[str] = []
    answer: str | None = None
    for _ in range(max_turns):
        text = chat.complete(messages, sample_index=sample_index)
        transcript.append(text)
        messages.append({"role": "assistant", "content": text})
        spans.append(Span(text, "llm"))
        action = parse_action(text)
        if action is None:
            rethink = "My action is not correct. Let me rethink."
            messages.append({"role": "user", "content": rethink})
            spans.append(Span(rethink, "env"))
            continue
        kind, payload = action
        if kind == "answer":
            answer = payload
            break
        passages = retriever.search(payload, topk=topk)
        info = f"<information>\n{format_passages(passages)}\n</information>"
        messages.append({"role": "user", "content": info})
        spans.append(Span(info, "retrieved"))
    em = exact_match(answer, task["golden_answers"])
    return SearchTrajectory(
        task_id=task["id"],
        sample_index=sample_index,
        spans=spans,
        answer=answer,
        reward=em,
        features=trajectory_features(spans, em),
        transcript=transcript,
    )


class ScriptedSearch:
    """One sample per group searches even when search_rate is low.

    It only emits a gold answer after that string is already in the question
    or in retrieved text. The reward function does not read this shortcut.
    """

    def __init__(self, tasks: list[dict], group_n: int = 5):
        self.by_question = {task["question"]: task for task in tasks}
        self.group_n = group_n

    def complete(self, messages: list[dict], *, sample_index: int = 0) -> str:
        system = messages[0]["content"]
        match = re.search(r"search_rate=([+-]?\d+(?:\.\d+)?)", system)
        if match is None:
            raise RuntimeError("系统提示里没有 search_rate")
        search_rate = float(match.group(1))
        user = next(message["content"] for message in messages if message["role"] == "user")
        question_match = re.search(r"Question:\s*(.+)\s*$", user)
        if question_match is None:
            raise RuntimeError("题面里没有 Question")
        question = question_match.group(1).strip()
        task = self.by_question[question]
        seen = "\n".join(message["content"] for message in messages)
        if contains_answer(question, task["golden_answers"]) or contains_answer(seen, task["golden_answers"]):
            gold = task["golden_answers"][0]
            return f"<think>答案已经出现。</think>\n<answer> {gold} </answer>"
        draw = (sample_index + 1) / (self.group_n + 1)
        explore = 0.5 if sample_index % self.group_n == 0 else 0.0
        if search_rate + explore > draw:
            return f"<think>题面里没有手册结论。</think>\n<search>{question}</search>"
        return "<think>没有检索，直接猜一个常见抖动。</think>\n<answer> network_blip </answer>"
