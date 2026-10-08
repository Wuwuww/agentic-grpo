"""Local runbook retriever. Search-R1 treats the search engine as the environment."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for piece in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return tokens


def load_passages(path: Path | None = None) -> list[dict]:
    file = path or ROOT / "data" / "runbooks.json"
    return json.loads(file.read_text(encoding="utf-8"))


def load_questions(path: Path | None = None) -> list[dict]:
    file = path or ROOT / "data" / "search_qa.json"
    return json.loads(file.read_text(encoding="utf-8"))


class Retriever:
    """IDF-weighted overlap. Top-k passages are inserted as <information>."""

    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.doc_tokens = [set(tokenize(f"{item['title']} {item['text']}")) for item in passages]
        docs = len(passages) or 1
        df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        self.idf = {token: math.log((docs + 1) / (count + 0.5)) for token, count in df.items()}

    def search(self, query: str, topk: int = 3) -> list[dict]:
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self.doc_tokens):
            score = sum(self.idf.get(token, 0.0) for token in query_tokens if token in tokens)
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [self.passages[index] for score, index in scored[:topk] if score > 0]
        return chosen


def format_passages(passages: list[dict]) -> str:
    lines = [f"Doc {index}(Title: {item['title']}) {item['text']}" for index, item in enumerate(passages, start=1)]
    return "\n".join(lines)
